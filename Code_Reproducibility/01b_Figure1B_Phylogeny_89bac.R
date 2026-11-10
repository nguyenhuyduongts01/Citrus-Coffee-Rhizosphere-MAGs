suppressPackageStartupMessages({
  library(ape); library(dplyr); library(tidyr); library(RColorBrewer)
})

ROOT_DIR <- if (dir.exists("Source_Data")) "." else ".."
FIG_DIR  <- file.path(ROOT_DIR, "Figures/Panels")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
out_file <- file.path(FIG_DIR, "Fig1B_Phylogeny_89MAGs_R.png")

tr   <- read.tree(file.path(ROOT_DIR, "Source_Data/results/Phylogenetic_Tree_89bac_MAGs.nwk"))
meta <- read.table(file.path(ROOT_DIR, "Source_Data/results/Metadata_89bac_MAGs.tsv"),
                   sep="\t", header=TRUE, stringsAsFactors=FALSE)

df <- data.frame(mag=tr$tip.label, stringsAsFactors=FALSE)
df <- left_join(df, meta %>% select(mag,completeness,contamination,length,phylum,host), by="mag")
df$completeness  <- ifelse(is.na(df$completeness),  75, df$completeness)
df$contamination <- ifelse(is.na(df$contamination),  0, df$contamination)
df$length        <- ifelse(is.na(df$length),          0, df$length)
df$host          <- ifelse(is.na(df$host), "Coffee", df$host)
df$phylum        <- ifelse(is.na(df$phylum)|df$phylum=="", "Unknown", df$phylum)

comp_pal <- colorRampPalette(c("#EFF3FF","#084594"))(101)
df$col_comp <- comp_pal[pmin(pmax(round(df$completeness),1),101)]
cont_pal <- colorRampPalette(c("#EDF8E9","#238B45"))(101)
max_cont <- max(df$contamination,na.rm=TRUE); if(max_cont==0) max_cont<-1
df$col_cont <- cont_pal[pmin(pmax(round((df$contamination/max_cont)*100)+1,1),101)]
host_col <- c(Citrus="#E53935", Coffee="#1E88E5")
df$col_host <- host_col[df$host]; df$col_host[is.na(df$col_host)] <- "#1E88E5"
df$col_bar  <- "#AAAAAA"

phyla_present <- unique(df$phylum[df$phylum!="Unknown"])
phy_abund_tsv <- file.path(ROOT_DIR, "Source_Data/results/MAG_Phylum_relative_abundance_89bac_12samples.tsv")
if (file.exists(phy_abund_tsv)) {
  phy_abund <- read.table(phy_abund_tsv, sep="\t", header=TRUE, row.names=1,
                          check.names=FALSE, stringsAsFactors=FALSE)
  phyla_ordered <- names(sort(colMeans(phy_abund, na.rm=TRUE), decreasing=TRUE))
  phyla <- c(phyla_ordered[phyla_ordered %in% phyla_present],
             sort(setdiff(phyla_present, phyla_ordered)))
} else {
  phyla <- sort(phyla_present)
}

# ── Load unified palette (same as Python stacked bar) ────────────────────────
pal_tsv <- file.path(ROOT_DIR, "Source_Data/results/Phylum_Color_Palette.tsv")
pal_df  <- read.table(pal_tsv, sep="\t", header=TRUE, stringsAsFactors=FALSE)
pal_map <- setNames(pal_df$color, pal_df$phylum)

# Build clade_pal: use unified colors where available, fallback for unknowns
fallback_cols <- c(RColorBrewer::brewer.pal(8,"Dark2"),
                   RColorBrewer::brewer.pal(8,"Set1"))
fb_i <- 1
clade_pal <- setNames(character(length(phyla)), phyla)
for (p in phyla) {
    if (p %in% names(pal_map)) {
        clade_pal[p] <- pal_map[p]
    } else {
        clade_pal[p] <- fallback_cols[(fb_i - 1) %% length(fallback_cols) + 1]
        fb_i <- fb_i + 1
    }
}
phylum_letters <- setNames(letters[1:length(phyla)], phyla)

sector_bounds <- function(theta_vec){
  ord<-order(theta_vec); th<-theta_vec[ord]; N<-length(th)
  diffs<-diff(th); mids<-th[-N]+diffs/2; step<-median(diffs)
  list(ord=ord,th=th,bounds=c(th[1]-step/2,mids,th[N]+step/2),step=step)
}

draw_ring <- function(ri,ro,col_vec,theta_vec){
  sb<-sector_bounds(theta_vec); col<-col_vec[sb$ord]; N<-length(sb$th)
  for(i in 1:N){ a<-seq(sb$bounds[i],sb$bounds[i+1],l=25)
    polygon(c(ri*cos(a),ro*cos(rev(a))),c(ri*sin(a),ro*sin(rev(a))),col=col[i],border=NA)}
  ab<-seq(min(sb$th)-sb$step/2,max(sb$th)+sb$step/2,l=1200)
  lines(ri*cos(ab),ri*sin(ab),col="#555555",lwd=0.5)
  lines(ro*cos(ab),ro*sin(ab),col="#555555",lwd=0.5)
}

draw_bars <- function(ri,heights,max_h,col_vec,theta_vec){
  sb<-sector_bounds(theta_vec); col<-col_vec[sb$ord]; h<-heights[sb$ord]; N<-length(sb$th)
  for(i in 1:N){ ro<-ri+(h[i]/max(heights,na.rm=TRUE))*max_h
    a<-seq(sb$bounds[i],sb$bounds[i+1],l=25)
    polygon(c(ri*cos(a),ro*cos(rev(a))),c(ri*sin(a),ro*sin(rev(a))),col=col[i],border=NA)}
}

draw_clades <- function(ri,ro,phylum_vec,theta_vec){
  sb<-sector_bounds(theta_vec); step<-sb$step
  # Place letters OUTSIDE the clade, in the gap between clade and rings
  label_r <- ro + 0.55 
  for(i in seq_along(phyla)){
    p<-phyla[i]; idx<-which(phylum_vec==p); if(!length(idx)) next
    th_sub<-theta_vec[idx]
    min_th<-min(th_sub)-step/2; max_th<-max(th_sub)+step/2
    angs<-seq(min_th,max_th,l=200)
    polygon(c(ri*cos(angs),ro*cos(rev(angs))),c(ri*sin(angs),ro*sin(rev(angs))),
            col=adjustcolor(clade_pal[p],0.40),border=NA)
    mid_th<-(min_th+max_th)/2
    deg<-mid_th*180/pi
    if(deg>90 & deg<270) deg<-deg+180
    plett <- phylum_letters[p]
    text(label_r*cos(mid_th), label_r*sin(mid_th), plett,
         srt=deg, adj=0.5, cex=2.0, font=2, col=clade_pal[p], xpd=TRUE)
  }
}

if(is.null(tr$edge.length)) tr$edge.length <- rep(1,nrow(tr$edge))
curr_depth <- max(node.depth.edgelength(tr))
tree_target <- 5.5
tr$edge.length <- tr$edge.length*(tree_target/curr_depth)

IMG_IN  <- 32; IMG_DPI <- 600; open_angle <- 10
png(out_file,width=IMG_IN,height=IMG_IN,units="in",res=IMG_DPI,bg="white")
par(mar=c(0,0,0,0),oma=c(0,0,0,0))

R_lim <- 18
plot(tr,type="fan",open.angle=open_angle,plot=FALSE,
     x.lim=c(-R_lim,R_lim),y.lim=c(-R_lim,R_lim))
env<-get("last_plot.phylo",envir=.PlotPhyloEnv)
N_tips<-length(tr$tip.label)
xx<-env$xx[1:N_tips]; yy<-env$yy[1:N_tips]
theta<-atan2(yy,xx); theta<-ifelse(theta<0,theta+2*pi,theta)
R_max_tree<-max(sqrt(xx^2+yy^2))

r_clade_in <- 0.1
r_clade_out <- R_max_tree+0.4
ring_w  <- 0.65; ring_gap <- 0.12
r_host_in  <- r_clade_out + 1.2  # Increased gap to 1.2 to fit the letters
r_host_out <- r_host_in +ring_w
r_comp_in  <- r_host_out+ring_gap
r_comp_out <- r_comp_in +ring_w
r_cont_in  <- r_comp_out+ring_gap
r_cont_out <- r_cont_in +ring_w
r_bar_in   <- r_cont_out+ring_gap*1.5
h_bar_max  <- 2.20

plot(0,0,type="n",xlim=c(-R_lim,R_lim),ylim=c(-R_lim,R_lim),axes=FALSE,xlab="",ylab="")

draw_clades(r_clade_in,r_clade_out,df$phylum,theta)
draw_ring(r_host_in, r_host_out, df$col_host, theta)
draw_ring(r_comp_in, r_comp_out, df$col_comp, theta)
draw_ring(r_cont_in, r_cont_out, df$col_cont, theta)
for(v in c(0.25,0.5,0.75,1.0)){
  r_ref<-r_bar_in+h_bar_max*v; angs<-seq(0,2*pi,l=1200)
  lines(r_ref*cos(angs),r_ref*sin(angs),col="#BBBBBB",lty=2,lwd=0.6)
}
draw_bars(r_bar_in,df$length,h_bar_max,df$col_bar,theta)

par(new=TRUE)
plot(tr,type="fan",open.angle=open_angle,show.tip.label=FALSE,
     edge.width=1.0,edge.color="#222222",
     x.lim=c(-R_lim,R_lim),y.lim=c(-R_lim,R_lim))

lx<-R_lim*0.82; lce<-2.6

legend(lx,R_lim*0.99,
       legend=c("Citrus","Coffee"),
       fill=c("#E53935","#1E88E5"),border="white",
       title="Host Species",title.font=2,bty="n",cex=lce)

legend(lx,R_lim*0.72,
       legend=c("100%","75%","50%"),
       fill=c("#084594","#6BAED6","#EFF3FF"),border="white",
       title="Completeness (%)",title.font=2,bty="n",cex=lce)

legend(lx,R_lim*0.45,
       legend=c(sprintf("%.0f%%",max_cont),sprintf("%.0f%%",max_cont/2),"0%"),
       fill=c("#238B45","#74C476","#EDF8E9"),border="white",
       title="Contamination (%)",title.font=2,bty="n",cex=lce)

min_size <- min(df$length[df$length>0],na.rm=TRUE)
max_size <- max(df$length,na.rm=TRUE)
legend(lx,R_lim*0.18,
       legend=c(sprintf("Max: %.2f Mbp", max_size/1e6),
                sprintf("Min: %.2f Mbp", min_size/1e6)),
       lwd=c(7,3), col=c("#AAAAAA","#AAAAAA"),
       title="Genome Size",title.font=2,bty="n",cex=lce)

lx_left <- -19.0
phylum_legend_text <- paste0(phylum_letters, ": ", gsub("^p__","", names(phylum_letters)))

# Draw Bacteria text and scale bar next to Phylum legend
text(lx_left + 0.5, R_lim*0.93, "Bacteria", cex=3.5, font=2, col="#111111", xpd=TRUE, adj=0)

sb_len<-0.1*(R_max_tree/curr_depth*10)
sx0<- lx_left + 0.5; sx1<- sx0 + sb_len; sy<- R_lim*0.85
segments(sx0,sy,sx1,sy,lwd=4,col="#333333")
segments(sx0,sy-0.1,sx0,sy+0.1,lwd=2,col="#333333")
segments(sx1,sy-0.1,sx1,sy+0.1,lwd=2,col="#333333")
text((sx0+sx1)/2,sy-0.25,"0.1",cex=1.8,col="#333333", adj=0.5)

# Then draw Phylum legend below it
legend(lx_left, R_lim*0.75, legend=phylum_legend_text, fill=clade_pal,
       title="Phylum", title.font=2, bty="n", cex=lce*0.82, ncol=1)

dev.off()
cat("✓",out_file,"\n")
