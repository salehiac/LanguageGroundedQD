import numpy as np
import matplotlib.pyplot as plt
import cv2
from cairosvg import svg2png
import pdb
from lxml import etree
import functools
import string
import pprint

_colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

_colors_vec2_text={v:k for k,v in _colors.items()}


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
        
        tiled=tile_im(layout[10:layout.shape[0]-10,10:layout.shape[1]-10,:])
        self.tiling=layout.copy()
        self.tiling[10:layout.shape[0]-10, 10:layout.shape[1]-10,:]=tiled
        self.tiling[self.layout==0]=0

    def add_obj(self,im_path,name,pos,scale,change_back=False,reverse_rgb=False):
        """
        pos should be (x,y) (for horizontal, vertical)
        scale should be in (0,1]
        """
        if im_path[-4:]==".svg":
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

    def display(self,display_bbox,hold_on=False,path2d_info=None,save_to=""):

        layout=self.layout
        for name,_ in self.object_dict.items():
            layout=self.display_object(name,display=False,layout=layout,display_bbox=display_bbox)

        fig,axes=plt.subplots(1,2)
        #axes[0].imshow((0.7*layout+0.3*self.tiling).astype(np.uint8))
        axes[0].imshow(layout)
        
        if path2d_info is not None:
            path2d, real_w, real_h=path2d_info
            self.visualise_traj(path2d,real_w,real_h,hold_on=True,ax=axes[0])
        
        axes[1].imshow(self.tiling)

        assert not (hold_on and save_to), "hold_on is incompatible with save_to"

        if save_to:
            plt.savefig(save_to)
            plt.close()

        
        if not hold_on:
            plt.show()
            plt.close()

        return fig,axes[0]

    def annotate_traj(self, path2d_in:np.ndarray, real_w:float, real_h:float,step:int=1):
        """
        if step>1, then the trajectory is sampled at step intervals

        ATTENTION: the "pos" key in the output dict reverses the vertical axis
        """

        path2d=path2d_in.copy()
        path2d[:,0]=(path2d[:,0]/real_w)*self.layout.shape[1]
        path2d[:,1]=(path2d[:,1]/real_h)*self.layout.shape[0]

        pts=[pt for pt in path2d]
        annotations=list(map(lambda pt:(self.point_near_objects(pt[0],pt[1])),pts))
        annotations_colors=list(map(lambda pt:(self.get_area_info(int(pt[0]),int(pt[1]))),pts))

        pts_reverse_y_axis=[np.array([pt[0],200-pt[1]]) for pt in pts]
        res_lst=[{"timestep":timestep,"pos":p.tolist(),"semantics":a,"colors":c} for timestep,p,a,c in zip(range(path2d.shape[0]), pts_reverse_y_axis, annotations, annotations_colors)]

        if step<=1:  
            return res_lst
        else:
            summary=res_lst[::step]
            summary.append(res_lst[-1])
            return summary




    def visualise_traj(self, path2d_in:np.ndarray, real_w:float, real_h:float, hold_on:bool=True,ax=None):

        path2d=path2d_in.copy()
        path2d[:,0]=(path2d[:,0]/real_w)*self.layout.shape[1]
        path2d[:,1]=(path2d[:,1]/real_h)*self.layout.shape[0]

        if ax is not None:
            ax.plot(path2d[0,0],path2d[0,1],"ro")
            ax.plot(path2d[-1,0],path2d[-1,1],"ko")
            ax.plot(path2d[:,0],path2d[:,1],"b")
        else:
            plt.plot(path2d[0,0],path2d[0,1],"ro")
            plt.plot(path2d[-1,0],path2d[-1,1],"ko")
            plt.plot(path2d[:,0],path2d[:,1],"b")

        if not hold_on:
            plt.show()
            plt.close()

    def point_near_objects(self,x,y):
        """
        x is horizontal
        """

        thresh=30
        near_objects={}
        for name, kk in self.object_dict.items():
            pos,_,bbox=kk
            dist=np.linalg.norm(np.array(pos)-np.array([x,y]))
            #print(name,dist)
            if dist<thresh:

                offset=6
                if x<pos[0]-offset:
                    horizontal_dir="west"
                elif x>pos[0]+offset:
                    horizontal_dir="east"
                else:
                    horizontal_dir=""

                if y<pos[1]-offset:
                    vertical_dir="north"
                elif y>pos[1]+offset:
                    vertical_dir="south"
                else:
                    vertical_dir=""
               
                if vertical_dir or horizontal_dir:
                    near_objects[f"to the {vertical_dir} {horizontal_dir} of {name}"]=dist
                else:
                    near_objects[f"on {name}"]=dist

        return near_objects

    def get_area_info(self, x,y):
        """
        """

        c_in=_colors_vec2_text[tuple(self.tiling[y,x,:])]

        if c_in=="black":
            raise Exception("your point is on the wall, which should be inaccessible by the agent.")
        
        cur_col=self.tiling[y,x,:]
        def color_change(hh:int,ww:int,direction:str,increment=bool):

            if direction=="horizontal":
                ww_n=ww
                while (self.tiling[hh,ww_n]==cur_col).all():
                    ww_n=ww_n+1 if increment else ww_n-1
           
                return _colors_vec2_text[tuple(self.tiling[hh,ww_n])], ww_n
            
            if direction=="vertical":
                hh_n=hh
                while (self.tiling[hh_n,ww]==cur_col).all():
                    hh_n=hh_n+1 if increment else hh_n-1

                return _colors_vec2_text[tuple(self.tiling[hh_n,ww])], hh_n

        d_a={}
        d_a["right"]=color_change(hh=y,ww=x,direction="horizontal",increment=True)
        d_a["left"]=color_change(hh=y,ww=x,direction="horizontal",increment=False)
        d_a["down"]=color_change(hh=y,ww=x,direction="vertical",increment=True)
        d_a["up"]=color_change(hh=y,ww=x,direction="vertical",increment=False)
        

        return c_in, d_a


def read_svg_file(svg_file_path):
    
    png_image = svg2png(url=svg_file_path)
    nparr = np.frombuffer(png_image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def line_to_pt_dist_2d(m:np.ndarray,a:float,b:float,c:float):
    """
    returns dist from 2d point m to 2d line ax+by+c
    """
    assert m.ndim==1

    g=np.array([a,b])
    s=-c/b#intersection with y axis
    u=m-s

    h=u.transpose().dot(g)/np.linalg.norm(g)

    return h

def change_svg_background(file_name, output_name):
    
    svg_tree = etree.parse(file_name)
    root = svg_tree.getroot()

    if 'viewBox' in root.attrib:
        view_box = root.attrib['viewBox'].split()
        minx = float(view_box[0])
        miny = float(view_box[1])
        width = float(view_box[2])
        height = float(view_box[3])
    elif 'width' in root.attrib and 'height' in root.attrib:
        minx = 0
        miny = 0
        width = float(root.attrib['width'])
        height = float(root.attrib['height'])
    else:
        raise ValueError("SVG file must have either a viewBox attribute or both width and height attributes.")

    black_rectangle = etree.Element("{http://www.w3.org/2000/svg}rect",
                                    x=str(minx), y=str(miny), width=str(width),
                                    height=str(height), fill='white')

    root.insert(0, black_rectangle)

    etree.ElementTree(root).write(output_name, pretty_print=True)

def create_env_with_objects(ressources_path="."):
    x=cv2.imread(f"{ressources_path}/ressources/maze_19_2.pbm")
    scene=Scene(x)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/freedo-Cactus.svg",name="cactus",pos=(180,133),scale=0.03,change_back=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/table-fan_jh.svg",name="fan",pos=(60,60),scale=0.04,change_back=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/Rfc1394-Blue-Sofa.svg",name="sofa",pos=(167,20),scale=0.1,change_back=True,reverse_rgb=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/tan-bed.svg",name="bed",pos=(164,174),scale=0.3,change_back=True,reverse_rgb=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/fridge.png",name="fridge",pos=(22,172),scale=0.1,change_back=False)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/cabinet_wood.svg",name="cabinet",pos=(23,24),scale=0.016,change_back=True,reverse_rgb=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/officeChair.svg",name="chair",pos=(98,175),scale=0.05,change_back=True,reverse_rgb=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/statue.svg",name="statue",pos=(120,95),scale=0.3,change_back=True,reverse_rgb=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/file_cabinet.svg",name="file_cabinet",pos=(140,62),scale=0.15,change_back=True,reverse_rgb=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/bathtub.svg",name="bathtub",pos=(58,103),scale=0.086,change_back=True,reverse_rgb=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/table.svg",name="table",pos=(123,175),scale=0.2,change_back=True,reverse_rgb=True)
    scene.add_obj(f"{ressources_path}/ressources/openclip_vector_images/stove.svg",name="stove",pos=(64,175),scale=0.03,change_back=True,reverse_rgb=True)

    return scene



if __name__=="__main__":

    test_annotate_point=True
    #test_annotate_point=False

    #test_visualize_traj=True
    test_visualize_traj=False

    if test_annotate_point:
        scene=create_env_with_objects()
        fig,_=scene.display(display_bbox=False,hold_on=True)

        # Event handler function for mouse clicks
        def on_click(event):
            if event.button == 1:  # Left mouse button
                x = int(event.xdata)
                y = int(event.ydata)
                near_objects=scene.point_near_objects(x, y)
                print(f"objects near {x},{y}: {near_objects}")
                c_txt,d_a=scene.get_area_info(x,y)
                print(f"point is in the {c_txt} area, with neighbouring areas {d_a}")

        # Register the event handler function
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()

    if test_visualize_traj:

        scene=create_env_with_objects()

        traj_lst=["/tmp/path2d_2.npy","/tmp/path2d_0.npy","/tmp/path2d_11.npy"]
        for traj_path in traj_lst:
            traj=np.load(traj_path)
            
            annotation_lst=scene.annotate_traj(traj, real_h=600, real_w=600,step=200)
            pretty_print=pprint.PrettyPrinter(indent=4,sort_dicts=False)
            pretty_print.pprint(annotation_lst)
            
            fig,_=scene.display(display_bbox=False,hold_on=True,path2d_info=(traj,600,600))
            plt.show()





