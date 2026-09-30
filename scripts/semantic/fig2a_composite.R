# Figure 2A — niche summary, built to Fig2A_concept.svg
#
#   niche | abundance (bar) | prevalence (blocks, n of 15 specimens) | transcriptional
#   program | dot plot: ROS + signalling pathway scores
#
# Colour = mean logFC, dot size = significance (small n.s. / large FDR < 0.05), exactly as
# the concept. Inputs are CSVs top50_pathways.Rmd already writes -- no scoring happens here:
#   results/combined_signalling_oxphos_nrf2.csv
#   results/aggregated_filtered_coldata.csv
#
# SVG is written with svglite, never cairo's svg(): cairo composites through a filter +
# luminance masks that Affinity Publisher ignores, importing the figure as solid black.
#
#   Rscript scripts/semantic/fig2a_composite.R [results_dir]

suppressMessages({
  library(ggplot2); library(dplyr); library(readr); library(forcats); library(tidyr)
  library(patchwork)
})

RES <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else "results"
FDR <- 0.05
# hexbin object behind the old prevalence donuts (scripts/figures/niche_prev_density_code.r)
SF_HEX  <- "/vast/projects/BCRL_Multi_Omics/scripts/figures/sf_hex.rds"
# cell-level object behind the old density panel; only read when the density cache is absent
VEN_ALL <- "/vast/projects/BCRL_Multi_Omics/scripts/figures/ven_all.rds"

REN <- c(T_A="T-ROS", T_TC="T-PAN", T_P="T-P", T_AA="T-AC", T_AMN="T-MES", T_LE="T-OC",
         T_OM="T-GL", T_O="T-MET2", T_M="T-MET1", I="I", V="V", N="N")
NICHE_COLORS_OLD <- c(T_LE="#9FCF87", T_AMN="#94FFB5", T_P="#9068AC", V="#F77800",
                      T_A="#FFCC99", T_M="#2BCE48", I="#C39C94", N="#005C31",
                      T_TC="#0075DC", T_O="#F0A0FF", T_OM="#C20088", T_AA="#4C005C")
NCOL <- setNames(unname(NICHE_COLORS_OLD), unname(REN[names(NICHE_COLORS_OLD)]))

# the 8 tumour niches of §6-7, in the order the heatmaps read top to bottom
NORDER <- c("T-AC","T-PAN","T-MES","T-MET2","T-MET1","T-P","T-ROS","T-GL")

# Transcriptional programs are transcribed from Fig2A_concept.svg -- they are not in any
# results CSV, so check them against the semantic modules before publishing.
PROGRAM <- c("T-ROS"  = "mRNA processing, chromatin",
             "T-MES"  = "Cilium assembly, SUMOylation",
             "T-P"    = "Mitosis, G2M checkpoint",
             "T-GL"   = "mRNA splicing, chromatin",
             "T-AC"   = "Cholesterol, gap junctions",
             "T-PAN"  = "Synaptic transmission",
             "T-MET1" = "Synaptic vesicle clustering",
             "T-MET2" = "Catecholamine, serotonin uptake")

theme_set(theme_minimal(base_size = 11))
# every strip shares one y axis; only the first panel prints the niche names
bare <- theme(panel.grid = element_blank(),
              axis.title = element_blank(),
              axis.text.y = element_blank(),
              axis.ticks = element_blank(),
              plot.title = element_text(size = 10, hjust = 0, face = "plain"),
              plot.margin = margin(4, 6, 4, 6))

# ---- niche composition: abundance + prevalence over the 15 specimens ----------------
cd <- read_csv(file.path(RES, "aggregated_filtered_coldata.csv"), show_col_types = FALSE) %>%
  transmute(patient, niche = gsub("_", "-", niche), ncells) %>%
  group_by(patient, niche) %>% summarise(ncells = sum(ncells), .groups = "drop")

N_SPEC <- n_distinct(cd$patient)

comp <- cd %>%
  group_by(patient) %>% mutate(frac = ncells / sum(ncells)) %>% ungroup() %>%
  complete(patient, niche, fill = list(ncells = 0, frac = 0)) %>%
  filter(niche %in% NORDER) %>%
  group_by(niche) %>%
  summarise(abundance = mean(frac),           # absences count as 0 -> true abundance
            n_spec    = sum(ncells > 0), .groups = "drop") %>%
  mutate(nn = factor(niche, levels = NORDER))

# ---- 1. niche name + colour swatch --------------------------------------------------
p_name <- ggplot(comp, aes(0, fct_rev(nn))) +
  geom_tile(aes(fill = nn), width = 0.5, height = 0.5) +
  geom_text(aes(x = 0.45, label = nn), hjust = 0, size = 3.5) +
  scale_fill_manual(values = NCOL, guide = "none") +
  scale_x_continuous(limits = c(-0.35, 2.4), expand = c(0, 0)) +
  labs(title = "Niche") +
  bare + theme(axis.text.x = element_blank())

# ---- 2. abundance bar ---------------------------------------------------------------
p_ab <- ggplot(comp, aes(abundance, fct_rev(nn), fill = nn)) +
  geom_col(width = 0.62) +
  scale_fill_manual(values = NCOL, guide = "none") +
  scale_x_continuous(breaks = c(0, max(comp$abundance)), labels = c("0", "max"),
                     expand = expansion(mult = c(0, 0.04))) +
  labs(title = "Abundance", subtitle = "mean share of a specimen") +
  bare + theme(axis.text.x = element_text(size = 8, colour = "grey40"),
               plot.subtitle = element_text(size = 8, colour = "grey40"))

# ---- 3. prevalence blocks -------------------------------------------------------------
# Scored exactly as scripts/figures/niche_prev_density_code.r did for the donut version:
#   colSums(table(patient, niche) >= MIN_HEX) / n_patients
# i.e. a niche counts towards a patient only once it holds at least MIN_HEX hexbins there,
# which is why this is lower than "has any cells at all" -- a few stray hexbins no longer
# make a niche present. Same hexbin object as that script, so the numbers match the old
# figure; only the mark changes, donuts -> one block per patient.
MIN_HEX <- 40
NUM2OLD <- c("0"="T_LE","1"="T_AMN","2"="T_P","3"="V","4"="T_A","5"="T_M",
             "6"="I","7"="N","8"="T_TC","9"="T_O","10"="T_OM","11"="T_AA")

sf_hex <- readRDS(SF_HEX)
tb     <- table(sf_hex$patient, sf_hex$niche)
N_PAT  <- nrow(tb)                          # 26 sections, as in the old script
hits   <- colSums(tb >= MIN_HEX)

prev <- tibble(cluster = names(hits), n_pat = as.integer(hits)) %>%
  mutate(niche = unname(REN[NUM2OLD[cluster]]),        # cluster id -> old code -> new name
         prevalence = n_pat / N_PAT) %>%
  filter(!is.na(niche), niche %in% NORDER) %>%         # drops the "white" background class
  mutate(nn = factor(niche, levels = NORDER))
stopifnot(nrow(prev) == length(NORDER))

pv_blocks <- prev %>%
  tidyr::expand(nesting(nn, n_pat), slot = 1:N_PAT) %>%
  mutate(on = slot <= n_pat)

p_pv <- ggplot(pv_blocks, aes(slot, fct_rev(nn))) +
  geom_tile(aes(fill = ifelse(on, as.character(nn), NA_character_)),
            colour = "grey60", linewidth = 0.2, width = 0.52, height = 0.6) +
  scale_fill_manual(values = NCOL, na.value = "white", guide = "none") +
  scale_x_continuous(breaks = NULL, expand = expansion(add = 0.5)) +
  labs(title = "Prevalence",
       subtitle = sprintf("patients with ≥%d hexbins (n/%d)", MIN_HEX, N_PAT)) +
  bare + theme(plot.subtitle = element_text(size = 8, colour = "grey40"))

# ---- 3b. density ---------------------------------------------------------------------
# The other half of niche_prev_density_code.r: each hexbin goes to the niche holding most
# of its cells, and density is the mean cell count of that niche's hexbins. Drawn here as a
# bar in the niche palette rather than that script's viridis glow squares.
# Computing it needs ven_all.rds (1.1 GB, ~60 s), so the result is cached beside the figure.
DENS_CSV <- file.path(RES, "niche_hex_density.csv")
if (!file.exists(DENS_CSV)) {
  message("computing hexbin density from ", VEN_ALL, " (this loads 1.1 GB)")
  ven <- readRDS(VEN_ALL)
  htb <- table(ven$hexbin_id, ven$niche)
  hex_df <- data.frame(count = rowSums(htb), niche = colnames(htb)[max.col(htb)])
  d <- vapply(split(hex_df$count, hex_df$niche), mean, numeric(1))
  write_csv(tibble(cluster = names(d), density = as.numeric(d),
                   n_hexbins = as.integer(table(hex_df$niche)[names(d)])), DENS_CSV)
  rm(ven, htb); gc()
}

dens <- read_csv(DENS_CSV, show_col_types = FALSE) %>%
  # ven_all labels niches with the OLD hyphenated codes (T-A, T-AA, ...), not cluster ids
  mutate(niche = unname(REN[gsub("-", "_", cluster)])) %>%
  filter(!is.na(niche), niche %in% NORDER) %>%
  mutate(nn = factor(niche, levels = NORDER))
stopifnot(nrow(dens) == length(NORDER))

# Kept as the old script's glow balls, but uncoloured: an unfilled grey ring with a soft
# halo, both scaled by density. No viridis and no niche fill, so the column adds a third
# encoding without competing with the palette carrying identity elsewhere in the figure.
p_dn <- ggplot(dens, aes(1, fct_rev(nn))) +
  # one tight halo under the ring: wider or softer and the eight rows smear into a band.
  # Tinted with the niche colour, so the balls echo the swatch column without a second
  # palette -- the ring stays neutral and size still carries the number.
  ggshadow::geom_glowpoint(aes(size = density, shadowcolour = nn), shape = 16,
                           colour = "grey50", alpha = 0.01,
                           shadowalpha = 0.30, shadowsize = 1.0) +
  ggshadow::scale_shadowcolour_manual(values = NCOL, guide = "none") +
  geom_point(aes(size = density), shape = 21, fill = "#FFFFFF03", colour = "grey35",
             stroke = 0.7) +
  # spread over the OBSERVED range (9.1-13.2), not from zero: area-proportional sizing
  # makes eight balls that differ by <1.5x look identical. Read it as a ranking, not a ratio.
  scale_size(range = c(2.4, 4.4), name = "mean cells\nper hexbin",
             breaks = c(9, 11, 13), limits = c(9, 13.3)) +
  scale_x_continuous(limits = c(0.55, 1.45), breaks = NULL, expand = c(0, 0)) +
  labs(title = "Density", subtitle = "mean cells per hexbin") +
  bare + theme(plot.subtitle = element_text(size = 8, colour = "grey40"),
               legend.position = "bottom", legend.title = element_text(size = 8),
               legend.text = element_text(size = 7.5), legend.key.size = unit(0.6, "lines"))

# ---- 4. transcriptional program text ------------------------------------------------
p_pr <- ggplot(comp, aes(0, fct_rev(nn))) +
  geom_text(aes(label = PROGRAM[as.character(nn)]), hjust = 0, size = 3.3) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(title = "Transcriptional program") +
  bare + theme(axis.text.x = element_blank())

# ---- 5. pathway dot plot ------------------------------------------------------------
SORDER <- c("NOTCH","PDGF","RAS-MAPK","JAK-STAT","TGF-β","WNT","BMP")
ROSSET <- c("OXPHOS [GOBP]", "KEAP1-NFE2L2")

comb <- read_csv(file.path(RES, "combined_signalling_oxphos_nrf2.csv"), show_col_types = FALSE) %>%
  filter(niche %in% NORDER) %>%
  mutate(nn  = factor(niche, levels = NORDER),
         grp = factor(ifelse(set %in% ROSSET, "ROS", "Signalling pathways"),
                      levels = c("ROS", "Signalling pathways")),
         set = factor(set, levels = c(ROSSET, SORDER)),
         sig = FDR < !!FDR)
stopifnot(!any(is.na(comb$set)))

lim <- quantile(abs(comb$mean_logFC), 0.98, na.rm = TRUE)

p_dot <- ggplot(comb, aes(set, fct_rev(nn))) +
  geom_point(aes(fill = pmax(pmin(mean_logFC, lim), -lim), size = sig),
             shape = 21, colour = "grey45", stroke = 0.3) +
  facet_grid(. ~ grp, scales = "free_x", space = "free_x") +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
                       name = "Score", limits = c(-lim, lim),
                       # breaks pulled just inside the limits, else ggplot drops the end labels
                       breaks = c(-lim, lim) * 0.97, labels = c("low", "high"),
                       guide = guide_colourbar(display = "rectangles", barheight = 0.6,
                                               barwidth = 6, title.position = "top",
                                               direction = "horizontal")) +
  scale_size_manual(values = c(`FALSE` = 2.8, `TRUE` = 6.5),
                    labels = c(`FALSE` = "n.s.", `TRUE` = sprintf("FDR < %.2g", FDR)),
                    name = NULL) +
  guides(size = guide_legend(override.aes = list(fill = "white", colour = "grey45"))) +
  labs(x = NULL, y = NULL) +
  bare +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8.5),
        strip.text.x = element_text(size = 10),
        panel.grid.major = element_line(colour = "grey93"),
        legend.position = "bottom", legend.box = "horizontal",
        legend.title = element_text(size = 9), legend.text = element_text(size = 8))

# ---- assemble -----------------------------------------------------------------------
fig <- p_name + p_ab + p_dn + p_pv + p_pr + p_dot +
  plot_layout(nrow = 1, widths = c(0.62, 0.75, 0.6, 1.0, 1.15, 3.0)) +
  plot_annotation(title = "Figure 2A — niche summary",
                  theme = theme(plot.title = element_text(face = "bold", size = 13)))

ggsave(file.path(RES, "Fig2A.svg"), fig, device = svglite::svglite, width = 15, height = 6.0)
ggsave(file.path(RES, "Fig2A.png"), fig, width = 15, height = 6.0, dpi = 200)
cat("wrote", file.path(RES, c("Fig2A.svg", "Fig2A.png")), sep = "\n  "); cat("\n\n")

print(comp %>% left_join(prev, by = "niche") %>% left_join(dens, by = "niche") %>%
        arrange(match(niche, NORDER)) %>%
        transmute(niche, abundance = sprintf("%.1f%%", 100 * abundance),
                  density = round(density, 1),
                  prevalence = sprintf("%d/%d (%.0f%%)", n_pat, N_PAT, 100 * prevalence)))
