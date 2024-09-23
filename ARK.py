import cv2
import numpy as np
import torch

# Load your black image (replace 'black_image.png' with your actual image path)

def calculate_needleparams(image):
    # Find white pixels (assuming white is represented by pixel values > 0)
    Length = torch.zeros(1,image.shape[0])
    angle = torch.zeros(1,image.shape[0])
    cpu_image = image.clone()
    # cpu_image.detach().numpy()
    # print(cpu_image.shape)
    for i in range(cpu_image.shape[0]):
        Image = cpu_image[i,:,:].squeeze(0,1)
        # print(Image.shape)
        white_pixels = torch.argwhere(Image > 0.5)
        # print(white_pixels.shape[0])
        if white_pixels.shape[0]>0:
            # Get the leftmost, topmost, rightmost, and bottommost white pixels
            leftmost = white_pixels[white_pixels[:, 1].argmin()]  # Min x-coordinate
            topmost = white_pixels[white_pixels[:, 0].argmin()]   # Min y-coordinate
            rightmost = white_pixels[white_pixels[:, 1].argmax()] # Max x-coordinate
            bottommost = white_pixels[white_pixels[:, 0].argmax()]  # Max y-coordinate
            Length[0,i] = ((bottommost[1] - topmost[1])**2+(bottommost[0] - topmost[0])**2)**(1/2)
            angle[0,i] = 90-(torch.arctan2(bottommost[1] - topmost[1], bottommost[0] - topmost[0])*180/torch.pi)
            # print(f"Topmost: {topmost}, Bottommost: {bottommost}, Length: {Length}, Angle: {angle}")
        else:
            Length[0,i] = 0
            angle[0,i] = 0

    return Length, angle

if __name__ == "__main__":
    image = torch.rand((8, 1, 161, 161), dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)*0.6

    Length, angle = calculate_needleparams(image)
    print(Length, angle)

if __name__ == "__main__":
    image = torch.rand((8, 1, 161, 161), dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)*0.6

    Length, angle = calculate_needleparams(image)
    print(Length, angle)
