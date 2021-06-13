import os, shutil

src = "LitterImages/"
dest = "sorted"

files = os.listdir(src)

annos = [file for file in files if file[-4:]=='.xml']

root_annos = [file[:-4] for file in annos]

anno_images = [file for file in files if file[:-5] in root_annos]

for img, anno  in zip(anno_images, annos):
	shutil.copyfile(os.path.join(src, img),os.path.join(dest, 'images', img))
	shutil.copyfile(os.path.join(src, anno),os.path.join(dest, 'annotations', anno))