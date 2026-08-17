# data

Source documents the pipeline ingests. Not committed — see `.gitignore`.

Plain text first: PDF parsing and OCR are where these pipelines stall, so the
ingestion path is proven end to end on text before the PDF loader is pointed
at the same pipeline.

Locally this directory is the source. Once deployed, the loader reads from
object storage instead, and this becomes the development sample.
