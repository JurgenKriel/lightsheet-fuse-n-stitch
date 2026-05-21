#!/bin/bash
#SBATCH --job-name=ls_stage_czi
#SBATCH --partition=regular
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/scratch/users/kriel.j/output.%j.%N.log
#SBATCH --error=/vast/scratch/users/kriel.j/output.%j.%N.log

# Copy the CZI from stornext to /vast/scratch before running 07_direct_fuse.sh.
# Eliminates the stornext read bottleneck during parallel aicspylibczi subblock reads.

SRC="/stornext/Img/data/prkfs1/m/Microscopy/KylieLuong/Lightsheet/KL018_KL260427/KL018_85_D7_CT2AvIII_Overview.czi"
DST="/vast/scratch/users/kriel.j/KL018_lightsheet/KL018_85_D7_CT2AvIII_Overview.czi"

mkdir -p "$(dirname "$DST")"

echo "Staging CZI: $SRC → $DST"
echo "File size: $(du -sh "$SRC" 2>/dev/null | cut -f1)"
rsync --progress --inplace "$SRC" "$DST"
echo "Stage complete: $(du -sh "$DST" | cut -f1)"
