L_DEF_DIR=/nhc/Soroosh.Mani/sandbox/ondemand-storm-workflow/singularity/
L_IMG_DIR=/nhc/Soroosh.Mani/sandbox/ondemand-storm-workflow/singularity/imgs

mkdir -p $L_IMG_DIR
for i in info; do
    pushd $L_DEF_DIR/$i/
    sudo singularity build $L_IMG_DIR/$i.sif $i.def
#    singularity build --fakeroot $L_IMG_DIR/$i.sif $i.def
    popd
done
