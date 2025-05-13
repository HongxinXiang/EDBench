import mlcroissant as mlc
import json

# FileObjects and FileSets define the resources of the dataset.
distribution = [
    # gpt-3 is hosted on a GitHub repository:
    mlc.FileObject(
        id="github-repository",
        name="github-repository",
        description="OpenAI repository on GitHub.",
        content_url="https://github.com/openai/gpt-3",
        encoding_formats=["git+https"],
        sha256="main",
    ),
    # Within that repository, a FileSet lists all JSONL files:
    mlc.FileSet(
        id="jsonl-files",
        name="jsonl-files",
        description="JSONL files are hosted on the GitHub repository.",
        contained_in=["github-repository"],
        encoding_formats=["application/jsonlines"],
        includes="data/*.jsonl",
    ),
]
record_sets = [
    # RecordSets contains records in the dataset.
    mlc.RecordSet(
        id="jsonl",
        name="jsonl",
        # Each record has one or many fields...
        fields=[
            # Fields can be extracted from the FileObjects/FileSets.
            mlc.Field(
                id="jsonl/context",
                name="context",
                description="",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    # Extract the field from the column of a FileObject/FileSet:
                    extract=mlc.Extract(column="context"),
                ),
            ),
            mlc.Field(
                id="jsonl/completion",
                name="completion",
                description="The expected completion of the promt.",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(column="completion"),
                ),
            ),
            mlc.Field(
                id="jsonl/task",
                name="task",
                description=(
                    "The machine learning task appearing as the name of the"
                    " file."
                ),
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(
                        file_property=mlc._src.structure_graph.nodes.source.FileProperty.filename
                    ),
                    # Extract the field from a regex on the filename:
                    transforms=[mlc.Transform(regex="^(.*)\.jsonl$")],
                ),
            ),
        ],
    )
]

# Metadata contains information about the dataset.
metadata = mlc.Metadata(
    name="gpt-3",
    # Descriptions can contain plain text or markdown.
    description=(
        "Recent work has demonstrated substantial gains on many NLP tasks and"
        " benchmarks by pre-training on a large corpus of text followed by"
        " fine-tuning on a specific task. While typically task-agnostic in"
        " architecture, this method still requires task-specific fine-tuning"
        " datasets of thousands or tens of thousands of examples. By contrast,"
        " humans can generally perform a new language task from only a few"
        " examples or from simple instructions \u2013 something which current"
        " NLP systems still largely struggle to do. Here we show that scaling"
        " up language models greatly improves task-agnostic, few-shot"
        " performance, sometimes even reaching competitiveness with prior"
        " state-of-the-art fine-tuning approaches. Specifically, we train"
        " GPT-3, an autoregressive language model with 175 billion parameters,"
        " 10x more than any previous non-sparse language model, and test its"
        " performance in the few-shot setting. For all tasks, GPT-3 is applied"
        " without any gradient updates or fine-tuning, with tasks and few-shot"
        " demonstrations specified purely via text interaction with the model."
        " GPT-3 achieves strong performance on many NLP datasets, including"
        " translation, question-answering, and cloze tasks, as well as several"
        " tasks that require on-the-fly reasoning or domain adaptation, such as"
        " unscrambling words, using a novel word in a sentence, or performing"
        " 3-digit arithmetic. At the same time, we also identify some datasets"
        " where GPT-3's few-shot learning still struggles, as well as some"
        " datasets where GPT-3 faces methodological issues related to training"
        " on large web corpora. Finally, we find that GPT-3 can generate"
        " samples of news articles which human evaluators have difficulty"
        " distinguishing from articles written by humans. We discuss broader"
        " societal impacts of this finding and of GPT-3 in general."
    ),
    cite_as=(
        "@article{brown2020language, title={Language Models are Few-Shot"
        " Learners}, author={Tom B. Brown and Benjamin Mann and Nick Ryder and"
        " Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind"
        " Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and"
        " Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom"
        " Henighan and Rewon Child and Aditya Ramesh and Daniel M. Ziegler and"
        " Jeffrey Wu and Clemens Winter and Christopher Hesse and Mark Chen and"
        " Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and"
        " Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford"
        " and Ilya Sutskever and Dario Amodei}, year={2020},"
        " eprint={2005.14165}, archivePrefix={arXiv}, primaryClass={cs.CL} }"
    ),
    url="https://github.com/openai/gpt-3",
    distribution=distribution,
    record_sets=record_sets,
)

print(metadata.issues.report())
print(metadata)


# ############### write
# with open("croissant.json", "w") as f:
#   content = metadata.to_json()
#   content = json.dumps(content, indent=2)
#   print(content)
#   f.write(content)
#   f.write("\n")  # Terminate file with newline
#
# ############### read
dataset = mlc.Dataset(jsonld="croissant.json")
metadata = dataset.metadata.to_json()
records = dataset.records(record_set="jsonl")
print(f"{metadata['name']}: {metadata['description']}")

for i, record in enumerate(records):
  print(record)
  if i > 10:
    break
