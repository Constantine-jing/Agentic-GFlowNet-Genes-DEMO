# fetch_tcga.R
# Download a TCGA RNA-seq project, preprocess to log-scale, save three CSVs
# matching the contract in src/config.py.
#
# To switch projects: change `project` below, or pass it as CLI arg.
#
# Usage:
#   Rscript src/env/fetch_tcga.R                 # uses default below
#   Rscript src/env/fetch_tcga.R TCGA-LIHC       # override via CLI
#
# This script is idempotent — re-running uses the GDC cache in ./GDCdata/.

# ---- config ----
default_project <- "TCGA-BRCA"
top_n_genes     <- 2000    # keep most-variable genes after voom
min_count       <- 10      # per-gene filter before voom
# ----------------

args <- commandArgs(trailingOnly = TRUE)
project <- if (length(args) >= 1) args[1] else default_project

# Map project -> short slug used in output filenames (e.g. "TCGA-BRCA" -> "tcga_brca")
slug <- tolower(gsub("-", "_", project))

cat(sprintf("========================================\n"))
cat(sprintf(" fetch_tcga.R\n"))
cat(sprintf(" project : %s\n", project))
cat(sprintf(" slug    : %s\n", slug))
cat(sprintf("========================================\n"))

# ---- ensure packages ----
ensure_pkg <- function(pkg, bioc = FALSE) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (bioc) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager", repos = "https://cloud.r-project.org")
      }
      BiocManager::install(pkg, update = FALSE, ask = FALSE)
    } else {
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
  }
}
ensure_pkg("TCGAbiolinks", bioc = TRUE)
ensure_pkg("SummarizedExperiment", bioc = TRUE)
ensure_pkg("edgeR", bioc = TRUE)   # cpm + filterByExpr
ensure_pkg("limma", bioc = TRUE)   # voom
ensure_pkg("jsonlite")

suppressPackageStartupMessages({
  library(TCGAbiolinks)
  library(SummarizedExperiment)
  library(edgeR)
  library(limma)
  library(jsonlite)
})

# ---- locate repo root (script lives in src/env/) ----
this_file <- sub("--file=", "",
                 commandArgs(trailingOnly = FALSE)[grep("--file=",
                                                        commandArgs(trailingOnly = FALSE))])
repo_root <- normalizePath(file.path(dirname(this_file), "..", ".."))
data_dir  <- file.path(repo_root, "data")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

# Work from repo root so GDCdata/ cache lands in a predictable place.
setwd(repo_root)

# ---- 1. query ----
cat("[1/5] querying GDC...\n")
query <- GDCquery(
  project       = project,
  data.category = "Transcriptome Profiling",
  data.type     = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",
  sample.type   = c("Primary Tumor", "Solid Tissue Normal")
)

# ---- 2. download (cached) ----
cat("[2/5] downloading (cached in ./GDCdata/)...\n")
GDCdownload(query, method = "api", files.per.chunk = 20)

# ---- 3. prepare ----
cat("[3/5] preparing SummarizedExperiment...\n")
se <- GDCprepare(query, summarizedExperiment = TRUE)

counts <- assay(se, "unstranded")           # genes × samples, raw counts
sample_types <- as.character(colData(se)$sample_type)

# Normalize labels to match config.py schema: "normal" / "tumor"
group <- ifelse(sample_types == "Solid Tissue Normal", "normal",
         ifelse(sample_types == "Primary Tumor",       "tumor", NA))
keep_samples <- !is.na(group)
counts <- counts[, keep_samples]
group  <- group[keep_samples]

# Use symbol if available, else ensembl ID
gene_symbols <- rowData(se)$gene_name
gene_ids_raw <- rownames(counts)
gene_ids <- ifelse(!is.na(gene_symbols) & gene_symbols != "",
                   gene_symbols, gene_ids_raw)
# Deduplicate symbols (collapse by sum) — TCGA has a few
if (anyDuplicated(gene_ids)) {
  counts_df <- aggregate(counts, by = list(gene_id = gene_ids), FUN = sum)
  rownames(counts_df) <- counts_df$gene_id
  counts_df$gene_id <- NULL
  counts <- as.matrix(counts_df)
}

n_tumor  <- sum(group == "tumor")
n_normal <- sum(group == "normal")
cat(sprintf("   raw: %d genes × %d samples  (tumor=%d, normal=%d)\n",
            nrow(counts), ncol(counts), n_tumor, n_normal))

# ---- 4. filter + normalize ----
cat("[4/5] filtering low-count genes + voom normalization...\n")
dge <- DGEList(counts = counts, group = factor(group, levels = c("normal", "tumor")))
keep <- filterByExpr(dge, min.count = min_count)
dge <- dge[keep, , keep.lib.sizes = FALSE]
dge <- calcNormFactors(dge, method = "TMM")

design <- model.matrix(~ dge$samples$group)
v <- voom(dge, design, plot = FALSE)
logcpm <- v$E   # log2-CPM, genes × samples

# Keep top-N most-variable genes so the GFlowNet action space is tractable
gene_var <- apply(logcpm, 1, var)
top_idx  <- order(gene_var, decreasing = TRUE)[seq_len(min(top_n_genes, nrow(logcpm)))]
logcpm <- logcpm[top_idx, ]

cat(sprintf("   kept: %d genes × %d samples (top %d most variable)\n",
            nrow(logcpm), ncol(logcpm), top_n_genes))

# ---- 5. write outputs ----
cat("[5/5] writing outputs...\n")

# Clean sample IDs (TCGA barcodes have dashes; R prefers dots in colnames)
sample_ids <- colnames(logcpm)

expr_out   <- file.path(data_dir, sprintf("%s_rnaseq.csv", slug))
labels_out <- file.path(data_dir, sprintf("%s_labels.csv", slug))
meta_out   <- file.path(data_dir, sprintf("%s_meta.json", slug))

expr_df <- data.frame(gene_id = rownames(logcpm), logcpm, check.names = FALSE)
write.csv(expr_df, expr_out, row.names = FALSE)

labels_df <- data.frame(sample_id = sample_ids, group = group)
write.csv(labels_df, labels_out, row.names = FALSE)

meta <- list(
  project      = project,
  slug         = slug,
  n_genes      = nrow(logcpm),
  n_samples    = ncol(logcpm),
  n_tumor      = n_tumor,
  n_normal     = n_normal,
  top_n_genes  = top_n_genes,
  min_count    = min_count,
  normalization = "TMM + voom (log2-CPM)",
  fetched_at   = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
)
write(toJSON(meta, pretty = TRUE, auto_unbox = TRUE), meta_out)

cat("\n========================================\n")
cat(" ✅ done\n")
cat(sprintf("   %s\n", expr_out))
cat(sprintf("   %s\n", labels_out))
cat(sprintf("   %s\n", meta_out))
cat("========================================\n")
cat("\nNext: in src/config.py set   DATASET = \"", slug, "\"\n", sep = "")
cat("Then:  python -m src.env.reward\n")
