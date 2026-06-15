# Manas — v1.0.0 Rules of Operation

## Core Rules

Stay local for the local memory workflow.

Keep the user's taught memory on the user's machine.

Always preserve source metadata.

Never silently lose source memory.

Prefer safe fallback over panic.

Missing optional sidecars should not crash local answering.

Missing `brain.manas.sourceindex` is not fatal.

Corrupt `brain.manas.sourceindex` is not fatal.

Stale `brain.manas.sourceindex` is not fatal.

Corrupt or stale `brain.manas.sourceindex` falls back to `brain.manas.sources`.

Missing `brain.manas.sources` falls back to original file reread if possible.

Unknown questions return: `Not enough local memory to answer this yet.`

Manas should not invent facts beyond local evidence.

Manas should prefer source-backed answers.

Manas should show source paths when sources are available.

## Teaching Rules

`teach` is the primary high-level teaching command.

`teach` supports direct text input.

`teach` supports `.md` files.

`teach` supports `.txt` files.

`teach` supports folders that contain supported `.md` and `.txt` files.

`teach` should preserve source metadata.

`teach` should store source-backed memory when teaching files.

`teach` should train sequence memory from taught text.

`teach` can train transformer memory when `--train-transformer` is used.

Successful teaching updates core memory.

Successful teaching updates sequence memory.

Successful teaching can update transformer memory.

Successful teaching updates source memory.

Successful teaching updates the source index.

Repeated teaching of the same file with the same content should not duplicate source memory.

Teaching a changed file should update source chunks safely.

Teaching a folder should teach supported files independently.

Unsupported files should be skipped safely in folder teaching.

Empty files should be skipped safely.

## Dry-Run Rules

`teach --dry-run` previews teaching without saving changes.

`teach --dry-run` must not mutate `brain.manas`.

`teach --dry-run` must not mutate `brain.manas.seq`.

`teach --dry-run` must not mutate `brain.manas.transformer`.

`teach --dry-run` must not mutate `brain.manas.langmeta`.

`teach --dry-run` must not mutate `brain.manas.sources`.

`teach --dry-run` must not mutate `brain.manas.sourceindex`.

Dry-run should not create source memory.

Dry-run should not create a source index.

## Source Memory Rules

The source memory sidecar is:

```txt
brain.manas.sources
```

`brain.manas.sources` is the source of truth for persisted source chunks.

Source memory stores original chunk text.

Source memory stores normalized searchable text.

Source memory stores token strings for local retrieval.

Source memory stores source paths.

Source memory stores stable source fingerprints.

Source memory stores stable chunk fingerprints.

Source memory preserves the original source path in answer output.

Source memory should support answering after the original taught file is moved or deleted.

Corrupt source memory should not be silently overwritten.

For the question what is source memory, source memory is the persisted local evidence stored in `brain.manas.sources`.

## Source Index Rules

The source index sidecar is:

```txt
brain.manas.sourceindex
```

`brain.manas.sourceindex` is derived from `brain.manas.sources`.

`brain.manas.sourceindex` should not duplicate full chunk text.

`brain.manas.sourceindex` maps tokens to source and chunk references.

`brain.manas.sourceindex` can be rebuilt.

`brain.manas.sourceindex` is a derived cache.

`brain.manas.sourceindex` is disposable because source memory is the source of truth.

If `brain.manas.sourceindex` is missing, the answer path falls back to scanning `brain.manas.sources`.

If `brain.manas.sourceindex` is corrupt, the answer path falls back to scanning `brain.manas.sources`.

If `brain.manas.sourceindex` is stale, the answer path falls back to scanning `brain.manas.sources`.

For the question what is the source index, the source index is a derived token-to-source/chunk index for faster local retrieval.

For the question what happens if the source index is missing, Manas falls back to scanning `brain.manas.sources`.

## Answering Rules

`ask` answers questions from local source-backed memory.

`query --answer` uses the same local answer path as `ask`.

Normal `query` behavior remains separate from `query --answer`.

Answers should prefer local evidence.

Answers should show source paths when sources are available.

Answers should not invent facts beyond local evidence.

Answers should be short and source-backed.

The answer priority is:

```txt
ask / query --answer
1. Use source index if fresh
2. Fallback to source memory scan
3. Fallback to original source files if needed
4. Return no-answer if no evidence exists
```

When evidence is missing, Manas should say: `Not enough local memory to answer this yet.`

When evidence is weak, Manas should avoid unsupported claims.

## Storage Rules

`brain.manas` stores core neural memory.

`brain.manas.seq` stores sequence memory and token transitions.

`brain.manas.transformer` stores transformer weights.

`brain.manas.langmeta` stores language metadata.

`brain.manas.sources` stores AI-ready persisted source memory.

`brain.manas.sourceindex` stores the token-to-source/chunk inverted index.

`brain.manas.sources` should be preserved carefully.

`brain.manas.sourceindex` can be rebuilt from `brain.manas.sources`.

Source memory and source index are not the same thing.

The source memory is the source of truth.

The source index is a derived retrieval cache.

## Fallback Rules

Safe fallback is preferred over panic.

If the source index is unavailable, use source memory.

If source memory is unavailable, reread original source files if possible.

If no local evidence exists, return the no-answer message.

The no-answer message is: `Not enough local memory to answer this yet.`

Fallback behavior should keep answering stable.

Fallback behavior should not hide real data corruption when teaching.

## Safety Rules

Do not overclaim what Manas knows.

Do not claim Manas is a hosted LLM.

Do not claim Manas is a ChatGPT clone.

Do not claim Manas can answer without evidence.

Do not treat a missing derived index as data loss.

Do not treat source memory and source index as the same thing.

Do not silently overwrite corrupt source memory.

Do not make local answer behavior depend on cloud APIs.

## Release Rules

Manas v1.0.0 ships the Linux x86_64 binary first.

macOS users build Manas from source for now.

Windows users build Manas from source for now.

The Manas release process is tag based.

A Manas release is created by pushing a tag like `v1.0.0`.

GitHub Actions uploads `manas-linux-x86_64.tar.gz`.

Do not create the GitHub Release manually when the tag workflow is used.

## Non-Goals

Manas v1.0.0 is not a hosted LLM service.

Manas v1.0.0 is not a vector database product.

Manas v1.0.0 is not a PDF or DOCX ingestion release for `teach`.

Manas v1.0.0 is not a macOS binary release.

Manas v1.0.0 is not a Windows binary release.

Manas v1.0.0 is not a replacement for large hosted language models.

Manas v1.0.0 is a stable project release, but Manas is still experimental as an AI system.