"""Reject common medical image formats in the Git index without printing names."""

import subprocess

_SUFFIXES = (
    b".nii",
    b".nii.gz",
    b".dcm",
    b".dicom",
    b".ima",
    b".mha",
    b".mhd",
    b".nrrd",
    b".nhdr",
    b".mgz",
    b".mnc",
    b".hdr",
    b".img",
)


def main() -> int:
    paths = subprocess.check_output(["git", "ls-files", "-z"], timeout=30)
    count = sum(path.lower().endswith(_SUFFIXES) for path in paths.split(b"\0"))
    if count:
        print(f"Git indexes {count} prohibited medical image path(s).")
        print("Remove them from the index. Use in-code synthetic fixtures.")
    return int(count > 0)


if __name__ == "__main__":
    raise SystemExit(main())
