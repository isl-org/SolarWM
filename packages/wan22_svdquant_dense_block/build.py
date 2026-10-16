"""Build the Wan2.2 SVDQuant SYCL extension."""

from __future__ import annotations

import argparse

from task import TestWan22SvdquantDenseBlock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-arch", default="bmg-g31")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Build the reference SYCL extension instead of the tuned extension",
    )
    args = parser.parse_args()

    task = TestWan22SvdquantDenseBlock()
    artifacts = (
        task.build_reference(args.gpu_arch)
        if args.reference
        else task.build(args.gpu_arch)
    )
    for artifact in artifacts:
        print(artifact)


if __name__ == "__main__":
    main()
