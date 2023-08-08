'''
    this lines of code are very hard coded using only plain function-oriented programming 
    its hard to extend the following code and also reuse, you can reuse the following lines of code
    to understand how the following codes works actually as it is easier to understand...
'''
from flask import Flask, request, redirect, render_template, url_for, session
import os
# torch
import torch
import fastai
from fastai.vision.all import *
from PIL import Image
import PIL
import torchvision.transforms as transforms
from fastai.vision.core import *
import matplotlib.cm as cm

# Flask utils

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
'''
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/flasktutorial'
db.init_app(app)
'''

#upload configurations
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static/uploads')
UPLOAD_FOLDER1 = os.path.join(os.path.dirname(__file__), 'static/Segments')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}



learner = fastai.learner.load_learner("C:/Users/DELL/myproject/deforestation_classifier.pkl")
state_dict = learner.model.state_dict()
model = torch.nn.Sequential(*list(learner.model.children()))
model.load_state_dict(state_dict)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)

def label_func(x):
  return mask_path+'/'+(x.stem + x.suffix)



model1 = load_learner('C:/Users/DELL/myproject/model9.pkl')







classes = ['AnnualCrop',
            'Forest',
             'HerbaceousVegetation',
              'Highway',
               'Industrial',
                'Pasture',
                 'PermanentCrop',
                  'Residential',
                   'River',
                    'SeaLake']





def model_predict(img_path, model):
    transformer = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    # Load the image
    img = Image.open(img_path).convert('RGB')
    img = transformer(img).unsqueeze(0)

    # Use the model to make a prediction
    with torch.no_grad():
        output = model(img)
    preds = torch.softmax(output, dim=1)

    return preds.argmax(dim=1)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    return render_template('index1.html')


@app.route("/classify", methods = ["POST", "GET"])
def classify():
    return render_template('appclassify.html')

@app.route('/classify_image', methods = ["POST", "GET"])
def classify_image():
    image2 = request.files.get('image2')
    image2.save(os.path.join(app.config['UPLOAD_FOLDER'], image2.filename))
    file_path2 = os.path.join(UPLOAD_FOLDER, image2.filename)
    preds = model_predict(file_path2, model)
    index_tensor = torch.from_numpy(preds.numpy()).unsqueeze(-1)
    class_names2 = [classes[index] for index in index_tensor]
    session["class_names2"] = class_names2
    return redirect(url_for('classify_display', filename2 = image2.filename))

@app.route('/classify_display/<filename2>')
def classify_display(filename2):
    class_name2 = session.get("class_names2", None)
    class_name2 = class_name2[0]
    return render_template('appclassifydisplay.html', 
                            image2 = url_for('static', filename='uploads/' + filename2),
                            class_name2 = class_name2)



@app.route("/segment", methods = ['POST', 'GET'])
def segment():
    return render_template('appsegment.html')


@app.route('/segment_display', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image1 = request.files.get('image1')
        image2 = request.files.get('image2')

        date1 = str(request.form['PastDate'])
        date2 = str(request.form['RecentDate'])
        
        date1 = date1[:-6] + " " + date1[-5:]
        date2 = date2[:-6] + " " + date2[-5:]
        # Save the files to the uploads folder in the static directory
        image1.save(os.path.join(app.config['UPLOAD_FOLDER'], image1.filename))
        image2.save(os.path.join(app.config['UPLOAD_FOLDER'], image2.filename))

        file_path2 = os.path.join(UPLOAD_FOLDER, image2.filename)
        file_path1 = os.path.join(UPLOAD_FOLDER, image1.filename)
        
        # segmentation part
        predict = model1.predict(file_path1)
        print(file_path1)
        pred_mask = predict[0]
        cmap = cm.get_cmap('viridis', np.max(pred_mask.numpy()) + 1)
        pred_mask_img = PIL.Image.fromarray((cmap(pred_mask.numpy())*255).astype(np.uint8)).convert("RGB")
        filename3 = 'image3.png'
        save_path1 = os.path.join(UPLOAD_FOLDER1, filename3)
        pred_mask_img.save(save_path1)

        forest = np.sum(pred_mask.numpy() == 6)
        # Calculate the total area of the image
        total_area = predict[0].shape[0] * predict[0].shape[1]
        # Calculate the segmented area1
        forested_area1 = forest/ total_area * 100
        deforested_area1 = 100 - forested_area1

        predict1 = model1.predict(file_path2)
        print(file_path2)
        pred_mask1 = predict1[0]
        cmap = cm.get_cmap('viridis', np.max(pred_mask1.numpy()) + 1)
        pred_mask_img1 = PIL.Image.fromarray((cmap(pred_mask1.numpy())*255).astype(np.uint8)).convert("RGB")  
        filename4 = 'image4.png'
        save_path2 = os.path.join(UPLOAD_FOLDER1, filename4)
        pred_mask_img1.save(save_path2)

        forest = np.sum(pred_mask1.numpy()== 6)
        total_area = predict[0].shape[0] * predict[0].shape[1]
        forested_area2 = (forest/ total_area) * 100
        deforested_area2 = 100 - forested_area2
        value = deforested_area2 - deforested_area1
        if value >0:
            string = 'deforested'
        else:
            string = 'forested'
        value = round(value, 2)
        value = abs(value)
        session["filename1"] = image1.filename
        session["filename2"] = image2.filename
        session["filename3"] = filename3
        session["filename4"] = filename4
        session["date1"] = date1
        session["date2"] = date2
        session["value"] = value
        session["string"] = string

        return redirect(url_for('segment_display', filename1 = image1.filename, filename2 = image2.filename, filename3 = filename3, filename4 = filename4))
    return '''
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="image1">
        <input type="file" name="image2">
        <input type="submit" value="Upload">
    </form>
    '''



@app.route('/segment_images/<filename1>/<filename2>/<filename3>/<filename4>')
def segment_display(filename1, filename2, filename3, filename4):
    value = session.get("value", None)
    date1 = session.get("date1", None)
    date2 = session.get("date2", None)
    string = session.get("string", None)
    return render_template('appsegmentdisplay.html', image1 = url_for('static', filename='uploads/' + filename1), 
                                                   image2 = url_for('static', filename='uploads/' + filename2),
                                                   image3 = url_for('static', filename='Segments/' + filename3),
                                                   image4 = url_for('static', filename='Segments/' + filename4),
                                                   date1 = date1, date2 = date2,
                                                    value = value,
                                                    string = string
                                                   )


@app.route("/previous", methods = ["POST"])
def previous():
    return redirect(url_for('segment'))


@app.route("/previous1", methods = ["POST"])
def previous1():
    return redirect(url_for('classify'))


@app.route("/redirect", methods=["POST"])
def next_picture():
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug = True)