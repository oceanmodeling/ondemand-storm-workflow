L_DEF_DIR=/work2/noaa/nos-surge/smani/sandbox/ondemand-storm-workflow/singularity/
L_IMG_DIR=/work2/noaa/nos-surge/smani/sandbox/ondemand-storm-workflow/singularity/imgs

mkdir -p $L_IMG_DIR
for i in prep; do
    pushd $L_DEF_DIR/$i/
#    sudo singularity build $L_IMG_DIR/$i.sif $i.def
    singularity build --fakeroot $L_IMG_DIR/$i.sif $i.def
    popd
done
