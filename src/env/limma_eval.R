# limma_eval.R
# Evaluate a gene subset via differential expression using limma.
#
# Called from Python with 5 command-line arguments:
#   Rscript limma_eval.R <expr_csv> <labels_csv> <subset_csv> <out_csv> <ctrl_label>,<trt_label>
#
# reward.py supplies paths from src/config.py, so this script is dataset-agnostic.
#
# Score = (n_sig / subset_size) * mean |logFC|

suppressPackageStartupMessages(library(limma))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: Rscript limma_eval.R <expr> <labels> <subset> <out> <ctrl,trt>")
}
expr_path   <- args[1]
labels_path <- args[2]
subset_path <- args[3]
out_path    <- args[4]
group_spec  <- strsplit(args[5], ",")[[1]]
ctrl_label  <- group_spec[1]
trt_label   <- group_spec[2]

# --- load data ---
expr <- read.csv(expr_path, row.names = 1, check.names = FALSE)
labels <- read.csv(labels_path, stringsAsFactors = FALSE)
subset_df <- read.csv(subset_path, stringsAsFactors = FALSE)

stopifnot(all(labels$sample_id %in% colnames(expr)))
expr <- expr[, labels$sample_id]

# --- subset to requested genes ---
chosen <- subset_df$gene_id
chosen <- chosen[chosen %in% rownames(expr)]
if (length(chosen) < 2) {
  write.csv(
    data.frame(n_sig = 0, mean_abs_logfc = 0, score = 0),
    out_path, row.names = FALSE
  )
  quit(status = 0)
}
expr_sub <- as.matrix(expr[chosen, , drop = FALSE])

# --- limma DE: ctrl vs trt ---
group <- factor(labels$group, levels = c(ctrl_label, trt_label))
design <- model.matrix(~ group)
fit <- lmFit(expr_sub, design)
fit <- eBayes(fit)
tt <- topTable(fit, coef = 2, number = Inf, sort.by = "none")

n_sig <- sum(tt$adj.P.Val < 0.05, na.rm = TRUE)
mean_abs_logfc <- mean(abs(tt$logFC), na.rm = TRUE)
score <- (n_sig / length(chosen)) * mean_abs_logfc

write.csv(
  data.frame(n_sig = n_sig, mean_abs_logfc = mean_abs_logfc, score = score),
  out_path, row.names = FALSE
)
cat(sprintf("[limma] subset=%d n_sig=%d mean|logFC|=%.3f score=%.4f\n",
            length(chosen), n_sig, mean_abs_logfc, score))
