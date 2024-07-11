from setuptools import setup


def install_requires(fname="requirements.txt"):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


setup(
    name="fetmrqc_sr",
    version="0.0.1",
    packages=["fetmrqc_sr"],
    description="Quality control for fetal brain MRI",
    author="Thomas Sanchez",
    author_email="thomas.sanchez@unil.ch",
    entry_points={
        "console_scripts": [
            "srqc_segmentation = fetmrqc_sr.cli.compute_segmentation:main",
            "srqc_compute_iqms = fetmrqc_sr.cli.compute_iqms:main",
        ],
    },
)
