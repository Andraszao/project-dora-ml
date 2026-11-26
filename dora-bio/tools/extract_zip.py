"""Safe zip extraction for tools/fetch_ext.sh (standard library only; run with python3 -I).

Usage: python3 -I tools/extract_zip.py ARCHIVE DEST

Downloaded archives are untrusted, so every entry is checked before anything is written: no absolute paths, no ".."
components, no symbolic links or other special files, and every target must resolve inside DEST. Only then is the
archive extracted. Exits 1 with a message on the first unsafe entry.
"""
import os
import stat
import sys
import zipfile


def main(archive: str, dest: str) -> int:
    root = os.path.realpath(dest)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            mode = info.external_attr >> 16
            if name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/"):
                print(f"unsafe path in {archive}: {name}")
                return 1
            if mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                print(f"special file (link or device) in {archive}: {name}")
                return 1
            target = os.path.realpath(os.path.join(root, name))
            if target != root and not target.startswith(root + os.sep):
                print(f"entry escapes the destination in {archive}: {name}")
                return 1
        os.makedirs(root, exist_ok=True)
        zf.extractall(root)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
