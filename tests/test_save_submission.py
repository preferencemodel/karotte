import errno
import os
import stat
import sys
from pathlib import Path

import pytest

from karotte.save_submission import save_submission
from karotte.schemas.transcript import ScoringEvent
from karotte.student_misbehavior import StudentMisbehaviorError, misbehavior_scoring

save_submission_module = sys.modules["karotte.save_submission"]


@pytest.fixture
def source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    return src


@pytest.fixture
def dest(tmp_path: Path) -> Path:
    return tmp_path / "saved"


def test_copies_single_file(tmp_path: Path, dest: Path):
    src = tmp_path / "answer.txt"
    src.write_text("42")

    result = save_submission(src, dest)

    assert result == dest
    assert dest.read_text() == "42"
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600


def test_copies_directory_tree(source: Path, dest: Path):
    (source / "a.txt").write_text("a")
    (source / "sub").mkdir()
    (source / "sub" / "b.txt").write_text("b")
    (source / "empty").mkdir()

    save_submission(source, dest)

    assert (dest / "a.txt").read_text() == "a"
    assert (dest / "sub" / "b.txt").read_text() == "b"
    assert (dest / "empty").is_dir()
    assert stat.S_IMODE(dest.stat().st_mode) == 0o700
    assert stat.S_IMODE((dest / "sub").stat().st_mode) == 0o700


def test_copies_weird_filenames(source: Path, dest: Path):
    (source / "with space.txt").write_text("x")
    (source / "uni-cödé.txt").write_text("y")

    save_submission(source, dest)

    assert (dest / "with space.txt").read_text() == "x"
    assert (dest / "uni-cödé.txt").read_text() == "y"


@pytest.fixture
def submissions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    subs = tmp_path / "subs"
    monkeypatch.setattr(save_submission_module, "SUBMISSIONS_DIR", subs)
    return subs


def test_default_dest_creates_and_returns_directory(
    tmp_path: Path, submissions_dir: Path
):
    src = tmp_path / "answer.txt"
    src.write_text("42")

    result = save_submission(src)

    assert result.parent == submissions_dir
    assert stat.S_IMODE(result.stat().st_mode) == 0o700
    assert (result / "answer.txt").read_text() == "42"


@pytest.mark.usefixtures("submissions_dir")
def test_default_dest_copies_directory_as_its_basename(source: Path):
    (source / "a.txt").write_text("a")

    result = save_submission(source)

    assert (result / "src" / "a.txt").read_text() == "a"


@pytest.mark.usefixtures("submissions_dir")
def test_default_dest_is_unique_per_call(tmp_path: Path):
    src = tmp_path / "answer.txt"
    src.write_text("42")

    assert save_submission(src) != save_submission(src)


def test_default_dest_is_removed_on_failure(source: Path, submissions_dir: Path):
    os.mkfifo(source / "pipe")

    with pytest.raises(StudentMisbehaviorError):
        save_submission(source)

    assert list(submissions_dir.iterdir()) == []


def test_missing_source_returns_an_empty_dest(tmp_path: Path, dest: Path):
    assert save_submission(tmp_path / "nope", dest) == dest
    assert not dest.exists()


def test_missing_source_returns_a_root_only_dir_holding_nothing(
    tmp_path: Path, submissions_dir: Path
):
    saved = save_submission(tmp_path / "nope.txt")

    assert saved.parent == submissions_dir
    assert list(saved.iterdir()) == []
    assert not (saved / "nope.txt").exists()
    assert stat.S_IMODE(saved.stat().st_mode) == 0o700


def test_missing_parent_is_misbehavior(tmp_path: Path, dest: Path):
    with pytest.raises(StudentMisbehaviorError, match="does not exist"):
        save_submission(tmp_path / "no_dir" / "nope", dest)
    assert not dest.exists()


def test_missing_source_behind_a_symlinked_parent_is_misbehavior(
    tmp_path: Path, dest: Path
):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")

    with pytest.raises(StudentMisbehaviorError):
        save_submission(tmp_path / "link" / "nope.txt", dest)
    assert not dest.exists()


def test_dangling_symlink_is_still_misbehavior(tmp_path: Path, dest: Path):
    link = tmp_path / "link.txt"
    link.symlink_to(tmp_path / "gone.txt")

    with pytest.raises(StudentMisbehaviorError, match="symlink"):
        save_submission(link, dest)


def test_dest_exists_raises_file_exists_error(source: Path, dest: Path):
    dest.mkdir()

    with pytest.raises(FileExistsError):
        save_submission(source, dest)


def test_source_symlink_raises(tmp_path: Path, dest: Path):
    real = tmp_path / "real.txt"
    real.write_text("x")
    link = tmp_path / "link.txt"
    link.symlink_to(real)

    with pytest.raises(StudentMisbehaviorError, match="symlink"):
        save_submission(link, dest)


def test_symlink_inside_tree_raises(source: Path, dest: Path):
    (source / "escape").symlink_to("/etc/passwd")

    with pytest.raises(StudentMisbehaviorError, match="symlink"):
        save_submission(source, dest)


def test_undecodable_entry_name_scores_as_serializable_misbehavior(
    source: Path, dest: Path
):
    """A symlink whose name is not valid UTF-8 must not smuggle surrogates into
    scoring metadata, which cannot be serialized to JSON."""
    try:
        os.symlink("/etc/passwd", os.path.join(os.fsencode(source), b"bad\xff\xfelink"))
    except OSError:
        pytest.skip("filesystem rejects non-UTF-8 filenames")

    with pytest.raises(StudentMisbehaviorError) as excinfo:
        save_submission(source, dest)

    scoring = misbehavior_scoring(excinfo.value)
    assert scoring.metadata["misbehavior"] == r"bad\udcff\udcfelink is a symlink"
    _ = ScoringEvent(scoring=scoring).model_dump_json()


def test_allow_symlinks_recreates_link_without_following(source: Path, dest: Path):
    (source / "dangling").symlink_to("../does/not/exist")

    save_submission(source, dest, allow_symlinks=True)

    assert os.readlink(dest / "dangling") == "../does/not/exist"


def test_symlink_in_source_path_raises_even_when_allowed(tmp_path: Path, dest: Path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("a")
    (tmp_path / "alias").symlink_to(real)

    with pytest.raises(StudentMisbehaviorError, match="symlink"):
        save_submission(tmp_path / "alias" / "a.txt", dest, allow_symlinks=True)


def test_fifo_raises(source: Path, dest: Path):
    os.mkfifo(source / "pipe")

    with pytest.raises(StudentMisbehaviorError, match="not a regular file"):
        save_submission(source, dest)


def test_file_over_max_file_bytes_raises(source: Path, dest: Path):
    (source / "big.bin").write_bytes(b"x" * 100)

    with pytest.raises(StudentMisbehaviorError, match="too large"):
        save_submission(source, dest, max_file_bytes=10)


def test_huge_sparse_file_fails_fast(source: Path, dest: Path):
    with open(source / "sparse.bin", "wb") as f:
        f.seek(10 * 1024**3)
        f.truncate()

    with pytest.raises(StudentMisbehaviorError, match="too large"):
        save_submission(source, dest)


def test_total_bytes_cap(source: Path, dest: Path):
    (source / "a.bin").write_bytes(b"x" * 60)
    (source / "b.bin").write_bytes(b"x" * 60)

    with pytest.raises(StudentMisbehaviorError, match="total size"):
        save_submission(source, dest, max_total_bytes=100)


def test_max_entries(source: Path, dest: Path):
    for i in range(5):
        (source / f"f{i}.txt").write_text("x")

    with pytest.raises(StudentMisbehaviorError, match="entries"):
        save_submission(source, dest, max_entries=3)


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT])
def test_disk_exhaustion_is_misbehavior(
    source: Path, dest: Path, monkeypatch: pytest.MonkeyPatch, code: int
):
    (source / "a.bin").write_bytes(b"x" * 10)

    def fail_write(_fd: int, _data: object) -> int:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(os, "write", fail_write)

    with pytest.raises(StudentMisbehaviorError, match="cannot save a.bin"):
        save_submission(source, dest)

    assert not dest.exists()


def test_dest_name_collision_is_misbehavior(
    source: Path, dest: Path, monkeypatch: pytest.MonkeyPatch
):
    (source / "sub").mkdir()
    real_mkdir = os.mkdir

    def fake_mkdir(path: str | Path, mode: int = 0o777, **kwargs: int) -> None:
        if Path(path).name == "sub":
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
        real_mkdir(path, mode, **kwargs)

    monkeypatch.setattr(os, "mkdir", fake_mkdir)

    with pytest.raises(StudentMisbehaviorError, match="cannot save sub"):
        save_submission(source, dest)

    assert not dest.exists()


def test_sparse_file_stays_sparse(source: Path, dest: Path):
    apparent = 10 * 1024**2
    with open(source / "sparse.bin", "wb") as f:
        f.write(b"x")
        f.truncate(apparent)

    save_submission(source, dest)

    out = dest / "sparse.bin"
    assert out.stat().st_size == apparent
    assert out.read_bytes() == (source / "sparse.bin").read_bytes()
    if sys.platform == "linux":
        assert out.stat().st_blocks * 512 < 5 * 1024**2


def test_sparse_zeros_count_toward_total_cap(source: Path, dest: Path):
    with open(source / "sparse.bin", "wb") as f:
        f.truncate(8 * 1024**2)

    with pytest.raises(StudentMisbehaviorError, match="total size"):
        save_submission(
            source, dest, max_file_bytes=16 * 1024**2, max_total_bytes=4 * 1024**2
        )


def test_dest_path_over_path_max_raises(source: Path, dest: Path):
    fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for i in range(26):
            name = f"{i:03d}".ljust(200, "x")
            os.mkdir(name, 0o700, dir_fd=fd)
            child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child_fd
    finally:
        os.close(fd)

    with pytest.raises(StudentMisbehaviorError, match="too long"):
        save_submission(source, dest)

    assert not dest.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_unsearchable_dir_raises(source: Path, dest: Path):
    locked = source / "locked"
    locked.mkdir()
    (locked / "f.txt").write_text("x")
    locked.chmod(0o444)

    try:
        with pytest.raises(StudentMisbehaviorError, match="cannot stat"):
            save_submission(source, dest)
    finally:
        locked.chmod(0o700)

    assert not dest.exists()


def test_max_depth(source: Path, dest: Path):
    deep = source / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "f.txt").write_text("x")

    with pytest.raises(StudentMisbehaviorError, match="deep"):
        save_submission(source, dest, max_depth=2)


def test_partial_dest_is_removed_on_failure(source: Path, dest: Path):
    (source / "ok.txt").write_text("x")
    os.mkfifo(source / "zz_pipe")

    with pytest.raises(StudentMisbehaviorError):
        save_submission(source, dest)

    assert not dest.exists()
