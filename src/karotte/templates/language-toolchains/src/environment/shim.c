/* The confinement sandbox.py gives a Python run, for a compiled one.
 *
 * A compiled artifact has no interpreter to bootstrap through, and the filter
 * cannot go in the launch's preexec because it refuses execve and the next
 * thing the child does is exec the artifact. So it goes in a constructor of a
 * preloaded shared object, which the loader runs after every library is mapped
 * and before the artifact's own init_array and main.
 *
 * What that leaves open is .preinit_array, which the loader runs first;
 * environment/elf.py refuses an artifact carrying one.
 *
 * Built three times, weakest rule last:
 *
 * -DALLOW_MAPPED_CODE keeps every rule but lets an unwritable file-backed
 * mapping be executable, for a cell whose runtime maps its own code out of the
 * artifact — Dart's AOT runtime does, and dies at startup otherwise. What that
 * costs is dlopen of something already on disk; what it does not cost is a JIT,
 * since no page is writable and executable at once either way. It rests on the
 * run's read-only filesystem, which is what leaves no file to write first and
 * map after — sandbox_probe.c checks that in the same breath.
 *
 * -DALLOW_JIT drops the executable-page rules entirely, for a runtime that
 * compiles as it runs and cannot be told not to: HotSpot dies at startup even
 * with -Xint. Everything below is still refused, so what the JIT'ed code can
 * do is what any code here can do.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>

#if defined(__x86_64__)
#define AUDIT_ARCH_NATIVE AUDIT_ARCH_X86_64
#elif defined(__aarch64__)
#define AUDIT_ARCH_NATIVE AUDIT_ARCH_AARCH64
#else
#error "no syscall table for this architecture"
#endif

#define X32_SYSCALL_BIT 0x40000000

#define OFF_NR offsetof(struct seccomp_data, nr)
#define OFF_ARCH offsetof(struct seccomp_data, arch)
/* Low half of args[n]. Little-endian only, which is every build target. */
#define OFF_ARG(n) (offsetof(struct seccomp_data, args) + 8 * (n))

#define KILL BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS)
#define REFUSED BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM)
#define LOAD_NR BPF_STMT(BPF_LD | BPF_W | BPF_ABS, OFF_NR)

/* Refuse one syscall, falling through when it does not match. */
#define REFUSE(nr) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (nr), 0, 1), REFUSED

static struct sock_filter filter[] = {
    /* Any ABI but this machine's native one dies rather than being filtered: a
       64-bit artifact has no business on i386 or x32, and translating the rules
       twice is how the two copies end up disagreeing. */
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, OFF_ARCH),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_NATIVE, 1, 0),
    KILL,

    LOAD_NR,
#if defined(__x86_64__)
    BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, X32_SYSCALL_BIT, 0, 1),
    KILL,
#endif

    REFUSE(SYS_ptrace),
    REFUSE(SYS_execve),
    REFUSE(SYS_execveat),
    REFUSE(SYS_memfd_create),
    REFUSE(SYS_io_uring_setup),
    REFUSE(SYS_io_uring_enter),
    REFUSE(SYS_io_uring_register),
    REFUSE(SYS_shmget),
    REFUSE(SYS_shmat),

    /* prctl(PR_SET_DUMPABLE, ...) would hand back the address space. */
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_prctl, 0, 4),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, OFF_ARG(0)),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, PR_SET_DUMPABLE, 0, 1),
    REFUSED,
    LOAD_NR,

    /* The JIT build has no rule here at all: every page may become
       executable, and the ALLOW below is the whole of it. */
#ifndef ALLOW_JIT
#ifdef ALLOW_MAPPED_CODE
    /* An executable mapping has to come from a file and be unwritable, so what
       runs is bytes that were on disk before the run. Writable is half the rule
       because a private mapping's copy-on-write pages never reach the file. */
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_mmap, 0, 6),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, OFF_ARG(2)),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, PROT_EXEC, 0, 10),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, PROT_WRITE, 2, 0),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, OFF_ARG(3)),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, MAP_ANONYMOUS, 0, 7),
    REFUSED,

    /* mprotect cannot tell the two apart, so it stays closed to both. */
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_mprotect, 1, 0),
#else
    /* No page ever becomes executable, which rules out a JIT and dlopen of
       anything the loader has not already mapped. */
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_mmap, 2, 0),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_mprotect, 1, 0),
#endif
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_pkey_mprotect, 0, 4),
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, OFF_ARG(2)),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, PROT_EXEC, 0, 1),
    REFUSED,
    LOAD_NR,
#endif

    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
};

static void fail(const char *what) {
  /* No stdio: this runs before the artifact's own initialisers. fd 2 is
     already the launch's captured stderr pipe. */
  (void)!write(2, "sandbox shim: ", 14);
  (void)!write(2, what, strlen(what));
  (void)!write(2, " failed\n", 8);
  _exit(127);
}

__attribute__((constructor)) static void confine(void) {
  struct sock_fprog program = {
      .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
      .filter = filter,
  };

  if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) fail("PR_SET_DUMPABLE");
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) fail("PR_SET_NO_NEW_PRIVS");
  if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program, 0, 0) != 0)
    fail("PR_SET_SECCOMP");
}
