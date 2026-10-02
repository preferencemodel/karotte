import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from karotte.firecracker import artifacts
from karotte.firecracker.artifacts import (
    Archive,
    ArchiveFile,
    ArtifactError,
    ensure_archive,
    ensure_artifacts,
    firecracker_archive,
    kernel_archive,
)
from karotte.firecracker.helper import (
    HELPER_BASE_IMAGE,
    HelperError,
    check_mke2fs,
    helper_containerfile,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tgz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeDownloads:
    content: dict[str, bytes]

    def __init__(self, content: dict[str, bytes]) -> None:
        self.content = content
        self.urls: list[str] = []

    def __call__(self, url: str, dest: Path) -> str:
        self.urls.append(url)
        dest.write_bytes(self.content[url])
        return _sha(self.content[url])


def _fake_extract(members: dict[str, bytes]):
    def extract(_archive: Path, _spec: Archive, file: ArchiveFile, dest: Path) -> None:
        dest.write_bytes(members[file.member])

    return extract


def _archive(url: str, data: bytes, member: str, content: bytes) -> Archive:
    return Archive(
        url=url,
        sha256=_sha(data),
        compression="zst",
        files=(ArchiveFile(member=member, name="kernel", sha256=_sha(content)),),
    )


class TestPinnedArchives:
    @pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
    def test_firecracker_release(self, arch: str):
        archive = firecracker_archive(arch)  # pyright: ignore[reportArgumentType]
        assert archive.url == (
            "https://github.com/firecracker-microvm/firecracker/releases/download/"
            + f"v1.17.0/firecracker-v1.17.0-{arch}.tgz"
        )
        assert [f.member for f in archive.files] == [
            f"release-v1.17.0-{arch}/firecracker-v1.17.0-{arch}",
            f"release-v1.17.0-{arch}/jailer-v1.17.0-{arch}",
        ]
        assert all(f.executable for f in archive.files)

    @pytest.mark.parametrize(
        ("arch", "kata_arch"), [("x86_64", "amd64"), ("aarch64", "arm64")]
    )
    def test_kernel_from_kata_static(self, arch: str, kata_arch: str):
        archive = kernel_archive(arch)  # pyright: ignore[reportArgumentType]
        assert archive.url.endswith(f"/4.0.0/kata-static-4.0.0-{kata_arch}.tar.zst")
        assert archive.compression == "zst"
        (kernel,) = archive.files
        assert kernel.member == "./opt/kata/share/kata-containers/vmlinux-6.18.35-200"
        assert len(kernel.sha256) == 64

    def test_helper_image_is_pinned_by_digest(self):
        assert "@sha256:" in HELPER_BASE_IMAGE
        assert helper_containerfile().startswith(f"FROM {HELPER_BASE_IMAGE}\n")


class TestHelperMke2fs:
    def test_a_helper_that_reads_a_tar_passes(self):
        check_mke2fs("mke2fs 1.47.2 (1-Jan-2025)\ntar-ok\n", "helper")

    def test_a_mke2fs_without_tar_input_is_refused(self):
        with pytest.raises(HelperError, match=r"1\.47\.0; .* needs e2fsprogs 1\.47\.1"):
            check_mke2fs("mke2fs 1.47.0 (5-Feb-2023)\ntar-ok\n", "helper")

    def test_a_mke2fs_built_without_libarchive_is_refused(self):
        """Alpine 3.23 and later: a new enough version that can't read a tar."""
        with pytest.raises(HelperError, match="libarchive"):
            check_mke2fs("mke2fs 1.47.4 (6-Mar-2025)\n", "helper")

    def test_unreadable_output_is_refused(self):
        with pytest.raises(HelperError, match="Could not read"):
            check_mke2fs("sh: mke2fs: not found\n", "helper")


class TestEnsureArchive:
    def test_downloads_checks_and_installs(self, tmp_path: Path):
        tarball, kernel = b"tarball", b"kernel bytes"
        archive = _archive("https://x/kata.tar.zst", tarball, "./vmlinux", kernel)
        downloads = FakeDownloads({archive.url: tarball})

        (path,) = ensure_archive(
            archive,
            tmp_path,
            download=downloads,
            extract=_fake_extract({"./vmlinux": kernel}),
        )

        assert path == tmp_path / "kernel"
        assert path.read_bytes() == kernel
        assert path.stat().st_mode & 0o777 == 0o644
        # Nothing but the kernel is kept.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["kernel"]

    def test_cached_file_is_not_downloaded_again(self, tmp_path: Path):
        tarball, kernel = b"tarball", b"kernel bytes"
        archive = _archive("https://x/kata.tar.zst", tarball, "./vmlinux", kernel)
        downloads = FakeDownloads({archive.url: tarball})
        extract = _fake_extract({"./vmlinux": kernel})

        ensure_archive(archive, tmp_path, download=downloads, extract=extract)
        ensure_archive(archive, tmp_path, download=downloads, extract=extract)

        assert downloads.urls == [archive.url]

    def test_corrupt_cached_file_is_replaced(self, tmp_path: Path):
        tarball, kernel = b"tarball", b"kernel bytes"
        archive = _archive("https://x/kata.tar.zst", tarball, "./vmlinux", kernel)
        (tmp_path / "kernel").write_bytes(b"tampered")
        downloads = FakeDownloads({archive.url: tarball})

        (path,) = ensure_archive(
            archive,
            tmp_path,
            download=downloads,
            extract=_fake_extract({"./vmlinux": kernel}),
        )

        assert downloads.urls == [archive.url]
        assert path.read_bytes() == kernel

    def test_download_with_wrong_checksum_is_refused(self, tmp_path: Path):
        archive = _archive("https://x/kata.tar.zst", b"expected", "./vmlinux", b"k")
        downloads = FakeDownloads({archive.url: b"something else"})

        with pytest.raises(ArtifactError, match="SHA-256"):
            ensure_archive(
                archive,
                tmp_path,
                download=downloads,
                extract=_fake_extract({"./vmlinux": b"k"}),
            )

        assert list(tmp_path.iterdir()) == []

    def test_member_with_wrong_checksum_is_refused(self, tmp_path: Path):
        archive = _archive("https://x/kata.tar.zst", b"tarball", "./vmlinux", b"k")
        downloads = FakeDownloads({archive.url: b"tarball"})

        with pytest.raises(ArtifactError, match="vmlinux"):
            ensure_archive(
                archive,
                tmp_path,
                download=downloads,
                extract=_fake_extract({"./vmlinux": b"not the kernel"}),
            )

        assert list(tmp_path.iterdir()) == []

    def test_gz_members_are_extracted_in_process(self, tmp_path: Path):
        binary, jailer = b"\x7fELF firecracker", b"\x7fELF jailer"
        data = _tgz({"release/firecracker": binary, "release/jailer": jailer})
        archive = Archive(
            url="https://x/firecracker.tgz",
            sha256=_sha(data),
            compression="gz",
            files=(
                ArchiveFile("release/firecracker", "firecracker", _sha(binary), True),
                ArchiveFile("release/jailer", "jailer", _sha(jailer), True),
            ),
        )

        paths = ensure_archive(
            archive, tmp_path, download=FakeDownloads({archive.url: data})
        )

        assert [p.read_bytes() for p in paths] == [binary, jailer]
        assert all(p.stat().st_mode & 0o777 == 0o755 for p in paths)


class TestEnsureArtifacts:
    def test_unsupported_arch(self, tmp_path: Path):
        with pytest.raises(ArtifactError, match="x86_64 and aarch64"):
            ensure_artifacts(tmp_path, arch="riscv64")

    def test_fetches_firecracker_and_kernel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        binary, jailer, kernel = b"fc", b"jailer", b"vmlinux"
        tgz = _tgz(
            {
                "release-v1.17.0-x86_64/firecracker-v1.17.0-x86_64": binary,
                "release-v1.17.0-x86_64/jailer-v1.17.0-x86_64": jailer,
            }
        )
        monkeypatch.setitem(artifacts._FIRECRACKER_TGZ_SHA256, "x86_64", _sha(tgz))  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setitem(artifacts._FIRECRACKER_SHA256, "x86_64", _sha(binary))  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setitem(artifacts._JAILER_SHA256, "x86_64", _sha(jailer))  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setitem(artifacts._KATA_TARBALL_SHA256, "x86_64", _sha(b"kata"))  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setitem(artifacts._KERNEL_SHA256, "x86_64", _sha(kernel))  # pyright: ignore[reportPrivateUsage]
        fc_url = firecracker_archive("x86_64").url
        kata_url = kernel_archive("x86_64").url
        downloads = FakeDownloads({fc_url: tgz, kata_url: b"kata"})

        def extract(path: Path, spec: Archive, file: ArchiveFile, dest: Path) -> None:
            if spec.compression == "gz":
                artifacts.extract(path, spec, file, dest)
            else:
                dest.write_bytes(kernel)

        result = ensure_artifacts(
            tmp_path, arch="x86_64", download=downloads, extract=extract
        )

        assert result.firecracker.read_bytes() == binary
        assert result.jailer.read_bytes() == jailer
        assert result.kernel.name == "vmlinux-6.18.35-200-kata-4.0.0-x86_64"
        assert result.kernel.read_bytes() == kernel
        assert downloads.urls == [fc_url, kata_url]
