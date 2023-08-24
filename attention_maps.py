from torchcam.utils import overlay_mask
from torchcam.cams import SmoothGradCAMpp

# Define your model
model=resnet18.eval()
model.fc=torch.nn.Linear(in_features=512, out_features=3)

# Set your CAM extractor
cam_extractor = SmoothGradCAMpp(model)

dirpath1= 'COVID-19_Radiography_Dataset\test\COVID'

for filename in os.listdir(dirpath1)[0:3]:         
    target_path= join(dirpath1,filename)
    image = Image.open(target_path).convert('RGB')

    image1 = test_transform(image)

    cam_extractor = SmoothGradCAMpp(model)
    # Get your input
    img = image1
  
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map, mode='F'), alpha=0.5)


    # Visualize the raw CAM
    fig,ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].axis('off')

    ax[1].imshow(result)
    ax[1].axis('off')

dirpath2='COVID-19_Radiography_Dataset\test\Normal'

for filename in os.listdir(dirpath2)[0:3]:  
    target_path= join(dirpath2,filename)
    image = Image.open(target_path).convert('RGB')

    image1 = test_transform(image)

    cam_extractor = SmoothGradCAMpp(model)

    img = image1
   
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    out = model(input_tensor.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map, mode='F'), alpha=0.5)

    # Visualize the raw CAM
    fig,ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].axis('off')

    ax[1].imshow(result)
    ax[1].axis('off')
    
dirpath3='COVID-19_Radiography_Dataset\test\Viral Pneumonia'

for filename in os.listdir(dirpath3)[0:3]:         
    target_path= join(dirpath3,filename)
    image = Image.open(target_path).convert('RGB')

    image1 = test_transform(image)

    cam_extractor = SmoothGradCAMpp(model)
    img = image1
  
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    out = model(input_tensor.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map, mode='F'), alpha=0.5)

    # Visualize the raw CAM
    fig,ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].axis('off')

    ax[1].imshow(result)
    ax[1].axis('off')
