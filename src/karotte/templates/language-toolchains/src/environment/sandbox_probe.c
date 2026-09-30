/* What the shim is supposed to have taken away, checked from inside it — the
 * compiled-run counterpart of sandbox.py's `refusals()`. Launched the way a
 * graded run is launched, with the shim preloaded. Anything it prints is a
 * hole, and it exits non-zero when it found any.
 *
 * `--mapped-code` and `--jit` ask the questions the way the other two shims
 * answer them: a mapping of a file may be executable under the first, any page
 * at all under the second, and everything else about them is unchanged. Each
 * mode is checked in both directions — a rule that is gone where the cell's
 * runtime needs it is a hole too, since it means the run will not start. */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <unistd.h>

static int holes;

#define hole(...)                             \
  do {                                        \
    fprintf(stderr, "sandbox: " __VA_ARGS__); \
    fputc('\n', stderr);                      \
    holes++;                                  \
  } while (0)

static const char *WRITE_PROBES[] = {"/tmp", "/var/tmp", "/dev/shm", "."};

static void nothing_is_writable(void) {
  for (size_t i = 0; i < sizeof(WRITE_PROBES) / sizeof(*WRITE_PROBES); i++) {
    char probe[PATH_MAX];
    snprintf(probe, sizeof probe, "%s/.sandbox-write-probe", WRITE_PROBES[i]);
    int fd = open(probe, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) continue;
    close(fd);
    unlink(probe);
    hole("%s is writable", WRITE_PROBES[i]);
  }
}

static void no_other_program_starts(void) {
  char *const argv[] = {(char *)"sandbox-probe", NULL};
  char *const envp[] = {NULL};
  execve("/nonexistent/sandbox-probe", argv, envp);
  if (errno != EPERM) hole("execve answered %s, not EPERM", strerror(errno));
}

static void no_page_becomes_executable(int mapped_code, int jit) {
  void *page = mmap(NULL, 4096, PROT_READ | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (page != MAP_FAILED) {
    munmap(page, 4096);
    if (!jit) hole("mmap still hands out executable pages");
  } else if (jit) {
    hole("mmap hands out no executable page, which this cell's runtime needs");
  }

  /* The other half of a JIT: take an ordinary page and turn it into code. */
  void *data = mmap(NULL, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (data != MAP_FAILED) {
    int became_code = mprotect(data, 4096, PROT_READ | PROT_EXEC) == 0;
    if (became_code && !jit)
      hole("mprotect still turns a page into code");
    else if (!became_code && jit)
      hole("mprotect turns no page into code, which this cell's runtime needs");
    munmap(data, 4096);
  }

  /* Mapping a file executable is the one the mapped-code shim allows, and the
     one this binary is a handy file for. */
  int fd = open("/proc/self/exe", O_RDONLY);
  if (fd < 0) {
    hole("could not open this program to map it");
    return;
  }
  void *text = mmap(NULL, 4096, PROT_READ | PROT_EXEC, MAP_PRIVATE, fd, 0);
  close(fd);
  if (text != MAP_FAILED) {
    munmap(text, 4096);
    if (!mapped_code && !jit) hole("a file still maps as executable");
  } else if (mapped_code || jit) {
    hole("a file no longer maps as executable, which this cell's runtime needs");
  }

  /* A writable one is a JIT under a file's name: the copy-on-write pages of a
     private mapping are the process's own, and nothing reaches the file. */
  fd = open("/proc/self/exe", O_RDONLY);
  if (fd < 0) {
    hole("could not open this program to map it");
    return;
  }
  void *rwx = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE,
                   fd, 0);
  close(fd);
  if (rwx != MAP_FAILED) {
    munmap(rwx, 4096);
    if (!jit) hole("a file maps as writable and executable at once");
  } else if (jit) {
    hole("a file no longer maps as writable code, which this cell's runtime "
         "needs");
  }
}

/* The mapped-code shim lets a runtime file map executable (checked above); a
 * file where the submission is staged must not, or a submission maps its `$0`
 * PROT_EXEC and runs machine code carried past the end of the source. The
 * grader stages one file per noexec mount and names them, colon separated, in
 * KAROTTE_SUBMISSION_PROBE_FILE. A probe handed none asks nothing, which is the
 * build-time one, with no such mount to ask about. */
static void no_staged_file_maps_executable(void) {
  const char *named = getenv("KAROTTE_SUBMISSION_PROBE_FILE");
  if (named == NULL) return;

  char paths[PATH_MAX * 4];
  if (strlen(named) >= sizeof paths) {
    hole("KAROTTE_SUBMISSION_PROBE_FILE is too long to read");
    return;
  }
  strcpy(paths, named);

  for (char *rest = paths, *path; (path = strsep(&rest, ":")) != NULL;) {
    if (*path == '\0') continue;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
      hole("could not open the staged probe file %s", path);
      continue;
    }
    errno = 0;
    void *text = mmap(NULL, 4096, PROT_READ | PROT_EXEC, MAP_PRIVATE, fd, 0);
    close(fd);
    if (text != MAP_FAILED) {
      munmap(text, 4096);
      hole("%s maps as executable", path);
    } else if (errno != EPERM && errno != EACCES) {
      hole("mapping %s answered %s, not the noexec refusal", path,
           strerror(errno));
    }
  }
}

static void the_address_space_has_no_handle_on_itself(void) {
  if (prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0) hole("the process is still dumpable");
  if (prctl(PR_SET_DUMPABLE, 1, 0, 0, 0) == 0) hole("PR_SET_DUMPABLE still answers");
  if (ptrace(PTRACE_TRACEME, 0, NULL, NULL) == 0) hole("ptrace still answers");

  /* Every path in /proc that reaches this address space. They are one file to
     the kernel, but only naming them all finds out. */
  long pid = (long)getpid();
  long tid = syscall(SYS_gettid);
  char alias[5][PATH_MAX];
  snprintf(alias[0], PATH_MAX, "/proc/self/mem");
  snprintf(alias[1], PATH_MAX, "/proc/%ld/mem", pid);
  snprintf(alias[2], PATH_MAX, "/proc/thread-self/mem");
  snprintf(alias[3], PATH_MAX, "/proc/self/task/%ld/mem", tid);
  snprintf(alias[4], PATH_MAX, "/proc/%ld/task/%ld/mem", pid, tid);
  for (int i = 0; i < 5; i++) {
    int fd = open(alias[i], O_WRONLY);
    if (fd < 0) continue;
    close(fd);
    hole("%s opens for writing", alias[i]);
  }
}

int main(int argc, char **argv) {
  int mapped_code = argc > 1 && strcmp(argv[1], "--mapped-code") == 0;
  int jit = argc > 1 && strcmp(argv[1], "--jit") == 0;

  nothing_is_writable();
  no_other_program_starts();
  no_page_becomes_executable(mapped_code, jit);
  no_staged_file_maps_executable();
  the_address_space_has_no_handle_on_itself();
  return holes ? 1 : 0;
}
