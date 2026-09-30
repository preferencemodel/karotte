"""Build-time layer: the image must never ship credential material.

`check_credentials()` is the build-time gate; the Containerfile test asserts the
default template runs it after the last step that is handed a build secret.
"""

import json
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from karotte.check_credentials import (
    CredentialInImage,
    check_credentials,
    find_credentials,
    home_directories,
    relocated_stores,
)

TEMPLATES = Path(__file__).resolve().parent.parent / "src/karotte/templates"


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "root"


def write(path: Path, content: str = "secret") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestFindCredentials:
    def test_clean_home_is_clean(self, home: Path):
        write(home / ".bashrc", "export PS1='$ '")
        assert find_credentials([home]) == []

    def test_missing_home_is_clean(self, home: Path):
        assert find_credentials([home]) == []

    def test_finds_the_uv_credential_store(self, home: Path):
        token = write(
            home / ".local/share/uv/credentials/credentials.toml",
            '[[credential]]\nservice = "https://registry.example/"\npassword = "s3cret"\n',
        )
        assert find_credentials([home]) == [token]

    def test_finds_every_file_in_a_store_directory(self, home: Path):
        store = home / ".local/share/uv/credentials"
        write(store / "credentials.toml", 'password = "a"')
        write(store / "3859a629b26fda96/tokens.json", "eyJ")
        assert len(find_credentials([home])) == 2

    def test_ignores_empty_lock_files(self, home: Path):
        """uv leaves zero-byte `.lock` files beside the tokens; they carry nothing."""
        write(home / ".local/share/uv/credentials/credentials.toml.lock", "")
        assert find_credentials([home]) == []

    def test_finds_a_private_key_but_not_its_public_half(self, home: Path):
        private = write(home / ".ssh/id_ed25519", "-----BEGIN OPENSSH PRIVATE KEY-----")
        write(home / ".ssh/id_ed25519.pub", "ssh-ed25519 AAAA")
        assert find_credentials([home]) == [private]

    def test_finds_gcloud_application_default_credentials(self, home: Path):
        adc = write(
            home / ".config/gcloud/application_default_credentials.json",
            '{"refresh_token": "1//0g"}',
        )
        assert find_credentials([home]) == [adc]

    def test_ignores_an_unauthenticated_gcloud_config_directory(self, home: Path):
        """Any `gcloud` call creates these; installing the SDK is not a credential."""
        write(home / ".config/gcloud/configurations/config_default", "[core]\n")
        write(home / ".config/gcloud/.last_update_check.json", '{"last_update": 0}')
        write(home / ".config/gcloud/logs/2026.08.26/run.log", "INFO\n")
        assert find_credentials([home]) == []

    def test_ignores_a_keyring_config_naming_a_backend(self, home: Path):
        """uv's `keyring-provider = "subprocess"` needs one, so this is expected."""
        write(
            home / ".config/python_keyring/keyringrc.cfg",
            "[backend]\ndefault-keyring=keyrings.example.ExampleKeyring\n",
        )
        assert find_credentials([home]) == []

    def test_finds_the_keyring_plaintext_store(self, home: Path):
        store = write(
            home / ".local/share/python_keyring/keyring_pass.cfg",
            "[registry.example]\nuser = s3cret\n",
        )
        assert find_credentials([home]) == [store]

    def test_follows_a_symlink_to_a_secret(self, home: Path, tmp_path: Path):
        """Skipping symlinks would let a linked `.netrc` read clean; it still ships."""
        target = write(
            tmp_path / "elsewhere/netrc", "machine registry.example password hunter2"
        )
        link = home / ".netrc"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        assert find_credentials([home]) == [link]

    def test_a_broken_symlink_is_not_a_credential(self, home: Path):
        link = home / ".netrc"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(home / "does-not-exist")
        assert find_credentials([home]) == []

    def test_a_symlink_loop_terminates(self, home: Path):
        """A store directory linking back up the tree must not recurse forever."""
        store = home / ".local/share/uv/credentials"
        store.mkdir(parents=True)
        write(store / "tokens.json", '{"access_token": "a"}')
        (store / "loop").symlink_to(home)
        assert any(p.name == "tokens.json" for p in find_credentials([home]))

    @pytest.mark.parametrize(
        "relative",
        [".netrc", ".git-credentials", ".config/gh/hosts.yml", ".aws/credentials"],
    )
    def test_finds_each_always_credential_path(self, home: Path, relative: str):
        path = write(home / relative)
        assert find_credentials([home]) == [path]

    def test_searches_every_home(self, tmp_path: Path):
        root = write(tmp_path / "root/.netrc")
        student = write(tmp_path / "workdir/.git-credentials")
        found = find_credentials([tmp_path / "root", tmp_path / "workdir"])
        assert found == sorted([root, student])


class TestConfigFilesNeedAMarker:
    """A bare registry or helper config is fine and common; only an inline secret is not."""

    def test_ignores_an_npmrc_without_a_token(self, home: Path):
        write(home / ".npmrc", "registry=https://registry.npmjs.org/\n")
        assert find_credentials([home]) == []

    def test_finds_an_npmrc_with_a_token(self, home: Path):
        path = write(
            home / ".npmrc",
            "//registry.npmjs.org/:_authToken=npm_xxxxxxxx\n",
        )
        assert find_credentials([home]) == [path]

    def test_ignores_an_npmrc_that_reads_its_token_from_the_environment(
        self, home: Path
    ):
        """`${NPM_TOKEN}` is the correct npm idiom — the token is *not* in the file."""
        write(home / ".npmrc", "//registry.npmjs.org/:_authToken=${NPM_TOKEN}\n")
        assert find_credentials([home]) == []

    def test_ignores_a_comment_that_names_a_credential_variable(self, home: Path):
        write(
            home / ".config/pip/pip.conf",
            "# password comes from UV_INDEX_PRIVATE_PASSWORD\n[global]\n"
            + "index-url = https://pypi.org/simple\n",
        )
        assert find_credentials([home]) == []

    def test_ignores_an_empty_value(self, home: Path):
        write(home / ".pypirc", "[pypi]\nusername = __token__\npassword =\n")
        assert find_credentials([home]) == []

    def test_ignores_a_docker_config_with_only_a_credential_helper(self, home: Path):
        write(
            home / ".docker/config.json",
            json.dumps({"credHelpers": {"registry.example.com": "gcloud"}}),
        )
        assert find_credentials([home]) == []

    def test_finds_a_docker_config_with_an_inline_auth(self, home: Path):
        path = write(
            home / ".docker/config.json",
            json.dumps({"auths": {"index.docker.io": {"auth": "dXNlcjpwYXNz"}}}),
        )
        assert find_credentials([home]) == [path]

    def test_ignores_a_pip_conf_without_credentials(self, home: Path):
        write(
            home / ".config/pip/pip.conf",
            "[global]\nindex-url = https://pypi.org/simple\n",
        )
        assert find_credentials([home]) == []

    def test_finds_a_credential_smuggled_into_an_index_url(self, home: Path):
        path = write(
            home / ".config/pip/pip.conf",
            "[global]\nindex-url = https://user:s3cret@registry.example/simple\n",
        )
        assert find_credentials([home]) == [path]

    def test_a_username_alone_in_a_url_is_not_a_credential(self, home: Path):
        write(
            home / ".config/uv/uv.toml",
            '[[index]]\nurl = "https://user@registry.example/simple/"\n',
        )
        assert find_credentials([home]) == []


class TestRelocatedStores:
    def test_follows_uv_credentials_dir(self, tmp_path: Path):
        store = write(tmp_path / "elsewhere/credentials.toml", 'password = "a"')
        roots = relocated_stores({"UV_CREDENTIALS_DIR": str(tmp_path / "elsewhere")})
        assert find_credentials([], roots) == [store]

    def test_follows_xdg_data_home(self, tmp_path: Path):
        store = write(tmp_path / "data/uv/credentials/credentials.toml", "{}")
        roots = relocated_stores({"XDG_DATA_HOME": str(tmp_path / "data")})
        assert find_credentials([], roots) == [store]

    def test_no_relocation_by_default(self):
        assert relocated_stores({}) == []


class TestHomeDirectories:
    def test_includes_the_student_workdir(self, tmp_path: Path):
        (tmp_path / "workdir").mkdir()
        homes = home_directories({"STUDENT_WORKDIR": str(tmp_path / "workdir")})
        assert (tmp_path / "workdir").resolve() in homes

    def test_skips_paths_that_do_not_exist(self, tmp_path: Path):
        homes = home_directories({"STUDENT_WORKDIR": str(tmp_path / "absent")})
        assert (tmp_path / "absent") not in homes

    def test_deduplicates_aliases_of_the_same_directory(self, tmp_path: Path):
        (tmp_path / "workdir").mkdir()
        homes = home_directories(
            {
                "HOME": str(tmp_path / "workdir"),
                "STUDENT_WORKDIR": str(tmp_path / "workdir"),
                "KAROTTE_WORKDIR": str(tmp_path / "workdir" / "." / ""),
            }
        )
        assert homes.count((tmp_path / "workdir").resolve()) == 1


class TestCheckCredentials:
    def test_passes_on_a_clean_image(self, home: Path):
        write(home / ".bashrc", "")
        check_credentials([home], extra_roots=[])

    def test_raises_listing_every_offender(self, home: Path):
        write(home / ".local/share/uv/credentials/credentials.toml", "{}")
        write(home / ".netrc", "machine registry.example password hunter2")
        with pytest.raises(CredentialInImage) as excinfo:
            check_credentials([home], extra_roots=[])
        message = str(excinfo.value)
        assert "credentials.toml" in message
        assert ".netrc" in message

    def test_the_error_says_to_rotate(self, home: Path):
        write(home / ".netrc", "machine registry.example password hunter2")
        with pytest.raises(CredentialInImage, match="rotate"):
            check_credentials([home], extra_roots=[])


class TestContainerfileGatesCredentials:
    """The gate must run after every step that is handed a build secret."""

    @pytest.fixture(scope="class")
    def rendered(self) -> str:
        env = Environment(
            loader=FileSystemLoader(TEMPLATES), keep_trailing_newline=True
        )
        return env.get_template("default/Containerfile").render()

    def test_check_runs_after_the_last_build_secret(self, rendered: str):
        """A secret-using step after the gate writes a store the gate never sees."""
        lines = rendered.splitlines()
        check_line = next(i for i, ln in enumerate(lines) if "karotte check" in ln)
        last_secret = max(i for i, ln in enumerate(lines) if "mount=type=secret" in ln)
        assert check_line > last_secret, (
            "karotte check must run after the last `--mount=type=secret` step"
        )
