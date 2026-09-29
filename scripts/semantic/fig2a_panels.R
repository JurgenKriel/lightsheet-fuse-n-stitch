# Fig 2A panels
#
#   1. pathway dot plot   - the §7 combined signalling + OXPHOS/NRF2 panel of
#                           top50_pathways.Rmd redrawn as dots: colour = mean logFC,
#                           size = significance (small = n.s., large = FDR < fdr)
#   2. niche prevalence   - how much of each spatial sample each niche accounts for,
#                           plus how many samples / patients carry it at all
#
# Inputs are the CSVs top50_pathways.Rmd already writes, so this script does no scoring:
#   results/combined_signalling_oxphos_nrf2.csv   (§7)
#   results/aggregated_filtered_coldata.csv       (niche x sample cell counts)
#
# Output: SVG + PNG per panel in results/. SVG is written with svglite, never cairo's
# svg() -- cairo composites through a filter + luminance masks that Affinity Publisher
# ignores, importing the figure as solid black.
#
#   Rscript scripts/semantic/fig2a_panels.R [results_dir]

suppressMessages({
  library(ggplot2); library(dplyr); library(readr); library(forcats); library(tidyr)
})

RES <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else "results"
FDR <- 0.05

# ---- shared house style -------------------------------------------------------------
# project niche palette, keyed on the OLD codes -> remapped to the new names (as in the Rmd)
REN <- c(T_A="T-ROS", T_TC="T-PAN", T_P="T-P", T_AA="T-AC", T_AMN="T-MES", T_LE="T-OC",
         T_OM="T-GL", T_O="T-MET2", T_M="T-MET1", I="I", V="V", N="N")
NICHE_COLORS_OLD <- c(T_LE="#9FCF87", T_AMN="#94FFB5", T_P="#9068AC", V="#F77800",
                      T_A="#FFCC99", T_M="#2BCE48", I="#C39C94", N="#005C31",
                      T_TC="#0075DC", T_O="#F0A0FF", T_OM="#C20088", T_AA="#4C005C")
NCOL   <- setNames(unname(NICHE_COLORS_OLD), unname(REN[names(NICHE_COLORS_OLD)]))
NORDER <- c("V","I","T-AC","T-PAN","T-MES","T-MET2","T-MET1","T-P","T-ROS","T-GL","T-OC","N")
TUMOUR <- setdiff(NORDER, c("V", "I", "N", "T-OC"))   # the 8 niches of §6-7

theme_set(theme_minimal(base_size = 11) +
          theme(panel.grid.minor = element_blank()))

# svglite + a vector colourbar: no embedded bitmap, real text, opens in Affinity
save_fig <- function(p, stem, width, height) {
  ggsave(file.path(RES, paste0(stem, ".svg")), p, device = svglite::svglite,
         width = width, height = height)
  ggsave(file.path(RES, paste0(stem, ".png")), p, width = width, height = height, dpi = 200)
  cat("wrote", file.path(RES, paste0(stem, c(".svg", ".png"))), sep = "\n  ")
  cat("\n")
}

# ---- panel 1: pathway dot plot ------------------------------------------------------
comb <- read_csv(file.path(RES, "combined_signalling_oxphos_nrf2.csv"), show_col_types = FALSE) %>%
  mutate(nn    = factor(niche, levels = intersect(NORDER, unique(niche))),
         panel = factor(panel, levels = c("signalling", "OXPHOS / NRF2")),
         sig   = FDR < !!FDR)

# set order: signalling first in the Rmd's SORDER, then the OXPHOS/NRF2 pair
SORDER <- c("NOTCH","PDGF","RAS-MAPK","JAK-STAT","TGF-β","WNT","BMP",
            "OXPHOS [GOBP]","KEAP1-NFE2L2")
comb <- comb %>% mutate(set = factor(set, levels = intersect(SORDER, unique(set))))
stopifnot(!any(is.na(comb$set)), !any(is.na(comb$nn)))

lim <- quantile(abs(comb$mean_logFC), 0.98, na.rm = TRUE)

p_dot <- ggplot(comb, aes(set, fct_rev(nn))) +
  geom_point(aes(fill = pmax(pmin(mean_logFC, lim), -lim), size = sig),
             shape = 21, colour = "grey35", stroke = 0.3) +
  facet_grid(. ~ panel, scales = "free_x", space = "free_x") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
                       name = "mean logFC",
                       guide = guide_colourbar(display = "rectangles")) +
  scale_size_manual(values = c(`FALSE` = 3, `TRUE` = 7.5),
                    labels = c(`FALSE` = "n.s.", `TRUE` = sprintf("FDR < %.2g", FDR)),
                    name = NULL) +
  labs(x = NULL, y = NULL,
       title = "Signalling + OXPHOS / NRF2 — mean logFC",
       subtitle = sprintf("%d sets scored on final_fit.rds the same way | %d tumour niches",
                          n_distinct(comb$set), n_distinct(comb$nn))) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text.x = element_text(face = "bold"),
        panel.grid.major = element_line(colour = "grey92"))

save_fig(p_dot, "combined_signalling_oxphos_nrf2_lfc_dotplot", 9, 4.8)

# ---- panel 2: niche prevalence across spatial samples -------------------------------
cd <- read_csv(file.path(RES, "aggregated_filtered_coldata.csv"), show_col_types = FALSE) %>%
  transmute(sample, patient,
            niche = gsub("_", "-", niche),   # coldata uses T_ROS, the figures use T-ROS
            ncells) %>%
  filter(niche %in% NORDER) %>%
  # one sample can carry a niche in several rows; collapse before taking shares
  group_by(sample, patient, niche) %>% summarise(ncells = sum(ncells), .groups = "drop")

# cohort = the study a sample belongs to, read off the patient id
cohort <- function(p) dplyr::case_when(grepl("^ven", p) ~ "Venture",
                                       grepl("^LGG", p) ~ "LGG",
                                       grepl("^GX",  p) ~ "GX",
                                       TRUE             ~ "GL")

# share of each SAMPLE's cells, so big and small sections are comparable
shares <- cd %>%
  group_by(sample) %>% mutate(frac = ncells / sum(ncells)) %>% ungroup() %>%
  mutate(nn = factor(niche, levels = NORDER), coh = cohort(patient)) %>%
  # samples ordered by patient, patients grouped by cohort
  arrange(coh, patient, sample) %>%
  mutate(sample = factor(sample, levels = unique(sample)),
         coh = factor(coh, levels = c("Venture", "GL", "LGG", "GX")))

n_samp <- n_distinct(shares$sample); n_pat <- n_distinct(shares$patient)

p_prev <- ggplot(shares, aes(sample, fct_rev(nn))) +
  geom_point(aes(size = frac, colour = nn)) +
  facet_grid(. ~ coh, scales = "free_x", space = "free_x") +
  scale_colour_manual(values = NCOL, guide = "none") +
  scale_size_area(max_size = 6, labels = scales::percent,
                  name = "share of the sample's cells") +
  labs(x = NULL, y = NULL,
       title = "Niche prevalence across the spatial samples",
       subtitle = sprintf("%d samples from %d patients | dot area = that niche's share of the sample",
                          n_samp, n_pat)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 6),
        strip.text.x = element_text(face = "bold"),
        panel.grid.major = element_line(colour = "grey92"))

save_fig(p_prev, "niche_prevalence_across_samples", 13, 4.6)

# ---- panel 2b: the concept's summary -- abundance + prevalence per niche ------------
# prevalence counts a niche as present in a sample if it has any cells there.
# abundance averages over EVERY sample -- complete() puts the absences back as 0, else a
# niche confined to a handful of sections looks more abundant than one found everywhere.
summ <- shares %>%
  tidyr::complete(nn, sample, fill = list(frac = 0, ncells = 0)) %>%
  group_by(nn) %>%
  summarise(n_samples  = n_distinct(sample[ncells > 0]),
            n_patients = n_distinct(patient[ncells > 0]),
            mean_frac  = mean(frac), .groups = "drop") %>%
  mutate(prev_samples = n_samples / n_samp, prev_patients = n_patients / n_pat)

write_csv(summ, file.path(RES, "niche_prevalence_summary.csv"))

p_summ <- summ %>%
  select(nn, `mean share of a sample` = mean_frac,
         `samples carrying it` = prev_samples, `patients carrying it` = prev_patients) %>%
  pivot_longer(-nn, names_to = "measure", values_to = "value") %>%
  mutate(measure = factor(measure, levels = c("mean share of a sample",
                                              "samples carrying it", "patients carrying it"))) %>%
  ggplot(aes(value, fct_rev(nn), fill = nn)) +
  geom_col(width = 0.72) +
  facet_grid(. ~ measure, scales = "free_x") +
  scale_fill_manual(values = NCOL, guide = "none") +
  scale_x_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.06))) +
  labs(x = NULL, y = NULL,
       title = "Niche abundance and prevalence",
       subtitle = sprintf("abundance = mean share of a sample's cells | prevalence = fraction of the %d samples / %d patients carrying the niche",
                          n_samp, n_pat)) +
  theme(strip.text.x = element_text(face = "bold"),
        panel.grid.major.y = element_blank())

save_fig(p_summ, "niche_abundance_prevalence", 10, 4.2)

cat("\ntumour niches (§6-7 panels):", paste(TUMOUR, collapse = ", "), "\n")
print(summ %>% arrange(desc(mean_frac)) %>% mutate(across(where(is.numeric), ~ signif(.x, 3))))
