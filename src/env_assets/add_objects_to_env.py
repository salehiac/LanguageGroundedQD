import numpy as np
import matplotlib.pyplot as plt
import cv2
from cairosvg import svg2png
import pdb
from lxml import etree
import functools
import string

_colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "gray": (128, 128, 128),
    "brown": (165, 42, 42),
}


def tile_im(im_in, num_tiles=3):
    
    im=im_in.copy()
    height=im.shape[0]
    width=im.shape[1]

    h_size=height//num_tiles
    w_size=width//num_tiles

    color_names=list(_colors.keys())
    c_idx=0
    for t_i in range(num_tiles):
        for t_j in range(num_tiles):

            im[t_i*h_size:(t_i+1)*h_size, t_j*w_size:(t_j+1)*w_size,:]=_colors[color_names[c_idx]]
            print(t_i,t_j,color_names[c_idx])
            c_idx=(c_idx+1)%len(color_names)

    return im


class Bbox:

    def __init__(self,left,right,up,down):
        self.left=left
        self.right=right
        self.up=up
        self.down=down
    def inside(self, x,y):

        b=(x<right and x>left)
        a=(y<up and y>down)
        return a and b

class Scene:

    def __init__(self,layout):
        self.layout=layout
        self.object_dict={}
    def add_obj(self,im_path,name,pos,scale,change_back=False,reverse_rgb=False):
        """
        pos should be (x,y) (for horizontal, vertical)
        scale should be in (0,1]
        """
        if im_path[-4:]==".svg":
            print(f"reading {im_path}")
            if change_back:
                random_string=functools.reduce(lambda x,y:x+y,np.random.choice(list(string.ascii_lowercase)+list(string.digits),size=15).tolist(),"")
                tmp_out_path=f"/tmp/SVG_{random_string}.svg"
                change_svg_background(im_path,tmp_out_path)
                im=read_svg_file(tmp_out_path)
            else:
                im=read_svg_file(im_path)
        elif im_path[-4:]==".png":
            im = cv2.imread(im_path)

        if reverse_rgb:
            [r,g,b]=cv2.split(im)
            im=cv2.merge([b,g,r])
        im=cv2.resize(im, dsize=(int(im.shape[1]*scale),int(im.shape[0]*scale)), interpolation = cv2.INTER_AREA)

        p_h=im.shape[0]
        offset_h=p_h%2
        p_w=im.shape[1]
        offset_w=p_w%2
        bbox=Bbox(up=max(0,pos[1]-p_h//2),down=min(pos[1]+p_h//2+offset_h,self.layout.shape[0]),left=max(0,pos[0]-p_w//2),right=min(pos[0]+p_w//2+offset_w,self.layout.shape[1]))
        
        self.object_dict[name]=(pos,im,bbox)


    def display_object(self,name,display=False,layout=None,display_bbox=True):

        layout=self.layout if layout is None else layout
        pos,patch,bbox=self.object_dict[name]
        layout[bbox.up:bbox.down,bbox.left:bbox.right,:]=patch

        if display_bbox:
            layout=cv2.rectangle(layout, (bbox.left, bbox.down), (bbox.right, bbox.up), (255, 0, 0), 2)
        
        if display:
            plt.imshow(layout)
            plt.show()
            plt.close()
        else:
            return layout

    def display(self):

        layout=self.layout
        for name,_ in self.object_dict.items():
            layout=self.display_object(name,display=False,layout=layout)

        plt.imshow(layout)
        plt.show()



def read_svg_file(svg_file_path):
    # Convert SVG to PNG using CairoSVG
    png_image = svg2png(url=svg_file_path)
    
    # Create a numpy array from PNG binary data
    nparr = np.frombuffer(png_image, np.uint8)
    
    # Decode numpy array into OpenCV image (BGR format)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img


def change_svg_background(file_name, output_name):
    # Parse SVG file
    svg_tree = etree.parse(file_name)
    root = svg_tree.getroot()

    # SVG file parameters.
    if 'viewBox' in root.attrib:
        # Use the viewBox if available
        view_box = root.attrib['viewBox'].split()
        minx = float(view_box[0])
        miny = float(view_box[1])
        width = float(view_box[2])
        height = float(view_box[3])
    elif 'width' in root.attrib and 'height' in root.attrib:
        # Otherwise, use the width and height attributes if available
        minx = 0
        miny = 0
        width = float(root.attrib['width'])
        height = float(root.attrib['height'])
    else:
        raise ValueError("SVG file must have either a viewBox attribute or both width and height attributes.")

    # Create a new element: black rectangle
    # Assuming that the SVG namespace is "http://www.w3.org/2000/svg"
    black_rectangle = etree.Element("{http://www.w3.org/2000/svg}rect",
                                    x=str(minx), y=str(miny), width=str(width),
                                    height=str(height), fill='white')

    # Insert the black rectangle at the beginning
    root.insert(0, black_rectangle)

    # Write the output SVG file
    etree.ElementTree(root).write(output_name, pretty_print=True)

def create_env_with_objects():
    x=cv2.imread("maze_19_2.pbm")
    scene=Scene(x)
    scene.add_obj("./openclip_vector_images/freedo-Cactus.svg",name="cactus",pos=(180,133),scale=0.03,change_back=True)
    scene.add_obj("./openclip_vector_images/table-fan_jh.svg",name="fan",pos=(60,60),scale=0.04,change_back=True)
    scene.add_obj("./openclip_vector_images/Rfc1394-Blue-Sofa.svg",name="sofa",pos=(167,20),scale=0.1,change_back=True,reverse_rgb=True)
    scene.add_obj("./openclip_vector_images/tan-bed.svg",name="bed",pos=(164,174),scale=0.3,change_back=True,reverse_rgb=True)
    scene.add_obj("./openclip_vector_images/fridge.png",name="fridge",pos=(22,172),scale=0.1,change_back=False)
    scene.add_obj("./openclip_vector_images/cabinet_wood.svg",name="cabinet",pos=(23,24),scale=0.016,change_back=True,reverse_rgb=True)
    scene.add_obj("./openclip_vector_images/officeChair.svg",name="chair",pos=(98,175),scale=0.05,change_back=True,reverse_rgb=True)
    scene.add_obj("./openclip_vector_images/statue.svg",name="statue",pos=(120,95),scale=0.3,change_back=True,reverse_rgb=True)
    scene.add_obj("./openclip_vector_images/file_cabinet.svg",name="file_cabinet",pos=(140,62),scale=0.15,change_back=True,reverse_rgb=True)
    scene.add_obj("./openclip_vector_images/bathtub.svg",name="bathtub",pos=(58,103),scale=0.086,change_back=True,reverse_rgb=True)
    scene.add_obj("./openclip_vector_images/table.svg",name="table",pos=(123,175),scale=0.2,change_back=True,reverse_rgb=True)
    scene.add_obj("./openclip_vector_images/stove.svg",name="stove",pos=(64,175),scale=0.03,change_back=True,reverse_rgb=True)
    scene.display()

    return scene


if __name__=="__main__":

    #add_colors=False
    add_colors=True
    if add_colors:
        x=cv2.imread("maze_19_2.pbm")
        y=x[10:x.shape[0]-10,10:x.shape[1]-10,:]
        plt.imshow(y)
        plt.show()

        im_tiled=tile_im(y)
        x_tmp=x.copy()
        x_tmp[10:x.shape[0]-10, 10:x.shape[1]-10,:]=im_tiled
        x_tmp[x==0]=0
        plt.imshow(x_tmp)
        plt.show()
    if 1:
        scene=create_env_with_objects()

