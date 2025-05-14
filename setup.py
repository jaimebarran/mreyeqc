from setuptools import setup


def install_requires(fname="requirements.txt"):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


setup(
    name="fetmrqc_sr",
    version="0.1.2",
    packages=["fetmrqc_sr"],
    description="Quality control for fetal brain MRI",
    author="Thomas Sanchez",
    author_email="thomas.sanchez@unil.ch",
    install_requires=install_requires(),
    entry_points={
        "console_scripts": [
            #"srqc_list_bids_csv = fetmrqc_sr.cli.list_and_anon_bids:main",
            #"srqc_generate_index = fetmrqc_sr.cli.generate_index:main",
            #"srqc_generate_reports = fetmrqc_sr.cli.generate_reports:main",
            #"srqc_brain_extraction = fetmrqc_sr.cli.brain_extraction:main",

            "srqc_segmentation = fetmrqc_sr.cli.compute_segmentation:main",
            "srqc_compute_iqms = fetmrqc_sr.cli.compute_iqms:main",

            #"srqc_niftymic_qc = fetmrqc_sr.cli.qc_niftymic:main",
            #ß"srqc_ratings_to_csv = fetmrqc_sr.cli.ratings_to_csv:main",

            "srqc_inference = fetmrqc_sr.cli.inference:main",
            "srqc_training = fetmrqc_sr.cli.train_model:main",

            #"srqc_reports_pipeline = fetmrqc_sr.cli.run_reports_pipeline:main",
            #"srqc_inference_pipeline = fetmrqc_sr.cli.run_inference_pipeline:main",
        ],
    },
)
