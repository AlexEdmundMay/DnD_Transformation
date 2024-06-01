import numpy as np
import argparse
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

def interpolate_image(img_array, x, y, x_interpolate,y_interpolate):
    #Seperately Interpolate RGB Values
    interp_r = RegularGridInterpolator((y, x), img_array[:,:,0], bounds_error=False, fill_value=0)
    interp_g = RegularGridInterpolator((y, x), img_array[:,:,1], bounds_error=False, fill_value=0)
    interp_b = RegularGridInterpolator((y, x), img_array[:,:,2], bounds_error=False, fill_value=0)
    
    #Get Transformed RGB Values By Transforming Positions of Grid
    transformed_r = interp_r(np.dstack((y_interpolate,x_interpolate)))
    transformed_g = interp_g(np.dstack((y_interpolate,x_interpolate)))
    transformed_b = interp_b(np.dstack((y_interpolate,x_interpolate)))
    
    #Combine RGB Values    
    transformed_image_array = np.dstack((transformed_r,transformed_g,transformed_b))
    return transformed_image_array

def get_position_grids(img_array):
    x = np.arange(0, len(img_array[0]))
    y = np.arange(0, len(img_array))
    
    #Get x,y Indices For Each Pixel
    x_grid = np.broadcast_to(x, (len(img_array),len(x)))
    y_grid = np.broadcast_to(y, (len(img_array[0]),len(y))).T
    return x,y,x_grid,y_grid

def get_transformation(type="dnd",x_0 = 0, y_0=0):
    if type == "dnd":
        #DnD (Chebyshev) Transformation Dependent On Angle wrt The Observer at (x_0,y_0).
        phi_observer = lambda x,y: np.arctan2((x-x_0),(y-y_0))
        transformation = lambda x,y: np.maximum(np.abs(np.cos(phi_observer(x,y))),np.abs(np.sin(phi_observer(x,y))))
    else:
        print("That Transformation Has Not Yet Been Defined. So Far Only 'dnd' can be used")
    return transformation

def get_transformed_image(image_path, transformation_type, x0_fraction, y0_fraction):
    #Extract Data From Image
    with Image.open(image_path) as img:
        img_array = np.asarray(img)
    
    #Get Img Grids To Use In Transformation And Interpolation    
    x,y,x_grid,y_grid = get_position_grids(img_array)

    #Find x0 and y0 as number of pixels.
    x0 = int(len(img_array[0])*x0_fraction)
    y0 = int(len(img_array[0])*(1-y0_fraction))
    print("x0:",x0,"y0:",y0)

    #Find Transformation Value For Each Point In Space
    transformation = get_transformation(transformation_type,x_0=x0,y_0=y0)
    transformation_matrix = transformation(x,y_grid)
    
    #Transform Image using Linear Interpolation
    transformed_image_array = interpolate_image(img_array, x-x0, y-y0, (x_grid-x0)/transformation_matrix,(y_grid-y0)/transformation_matrix)

    #Open Array as Image and Display
    img_transformed = Image.fromarray(np.uint8(transformed_image_array))
    return img_transformed

def animate_path(image_file, transformation_type,x_path,y_path):
    #Loop Over Points On The Path
    image_array = []
    for i,(x0,y0) in enumerate(zip(x_path,y_path)):
        #Create Image For Each Point
        img_transformed = get_transformed_image(image_file, transformation_type, x0, y0)
        image_array.append(img_transformed)
        
    #Save As GIF Or Same File Type As Original Image
    if len(image_array) == 1:
            image_array[0].save(image_file.replace("Images","Output").replace(".","_warped."))
    else:
        image_array[0].save(image_file.replace("Images","Output").replace(".","_")+".gif", save_all=True, append_images=image_array[1:], duration=100, loop=0)
    
def main():
    #Parse Command Line
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str,
                        help='File path of the image you want to transform.')
    parser.add_argument('-t', '--transformation', type=str,
                        help='Transformation to apply to image: "dnd" for standard DnD transformation (Chebyshev).')
    parser.add_argument('--x0', nargs='*', default=[0],
                        help='Where to define the x-axis location as fraction of the image width (0 left, 1 right).')
    parser.add_argument('--y0', nargs='*', default=[0],
                        help='Where to define the y-axis location as fraction of the image height (0 bottom, 1 top).')
    args = parser.parse_args()
    
    #Get Transformed Image(s)
    x_path = np.array(args.x0).astype(float)
    y_path = np.array(args.y0).astype(float)
    animate_path(args.image, args.transformation,x_path,y_path)
        
if __name__ == "__main__":
    main()


