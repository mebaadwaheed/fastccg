# FCVS CLI Tool (`fcvs`)

The `.fcvs` (FastCCG Vector Store) file is a custom JSON-based format used by FastCCG to save and load the state of a RAG knowledge base. To make managing these files easier, FastCCG provides a dedicated command-line tool: `fcvs`.

This tool helps you inspect the contents of a vector store, validate its integrity, and convert it to a standard JSON file for easier viewing.

## Installation

The `fcvs` tool is automatically installed when you install the `fastccg` package:

```bash
pip install fastccg
```

## Commands

The `fcvs` tool has three main commands: `inspect`, `validate`, and `convert`.

### 1. `fcvs inspect`

This command provides a quick overview of an `.fcvs` file, including its size, the number of vectors it contains, and the dimensionality of those vectors.

**Usage:**

```bash
fcvs inspect [OPTIONS] FILENAME
```

**Example:**

```bash
$ fcvs inspect my_knowledge.fcvs

--- Inspecting my_knowledge.fcvs ---
                 File Metadata
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric            ┃ Value                    ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ File Size         │ 13,480 bytes (~13.16 KB) │
│ Number of Vectors │ 3                        │
│ Vector Dimensions │ 128                      │
│ Metadata Present  │ True                     │
└───────────────────┴──────────────────────────┘
```

### 2. `fcvs validate`

This command checks if an `.fcvs` file is well-formed. It verifies that the file is valid JSON and that its internal structure matches the expected format (i.e., a dictionary where each value is a `[vector, metadata]` pair).

**Usage:**

```bash
fcvs validate [OPTIONS] FILENAME
```

**Example:**

```bash
$ fcvs validate my_knowledge.fcvs

--- Validating my_knowledge.fcvs ---
Success: File is a valid .fcvs file.
```

### 3. `fcvs convert`

This command converts an `.fcvs` file into a standard, human-readable `.json` file. Since `.fcvs` files are already JSON, this command essentially renames the file and pretty-prints the content with indentation.

**Usage:**

```bash
fcvs convert [OPTIONS] FILENAME
```

**Options:**

*   `--to TEXT`: The format to convert to. Currently, only `json` is supported.

**Example:**

```bash
$ fcvs convert my_knowledge.fcvs --to json

--- Converting my_knowledge.fcvs to JSON ---
Success: File converted and saved to 'my_knowledge.json'.
```

This will create a new file named `my_knowledge.json` in the same directory.
