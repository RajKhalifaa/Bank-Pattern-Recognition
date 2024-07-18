import numpy as np
import cv2 as cv
import h5py
import pickle
import logging
from tkinter import filedialog
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

logging.basicConfig(level=logging.INFO)

# -------------------- Load the file --------------------------------------------
def load_models():
    try:
        model_knn = pickle.load(open('model_KNN.p', 'rb'))
        model_mlp = pickle.load(open('model_MLP.p', 'rb'))
        model_svm = pickle.load(open('model_SVM.p', 'rb'))
        return model_knn, model_mlp, model_svm
    except FileNotFoundError as e:
        logging.error(f"Error loading model: {e}")
        return None, None, None

model_knn, model_mlp, model_svm = load_models()

if model_knn is None or model_mlp is None or model_svm is None:
    logging.error("Error loading models. Exiting...")
    exit()

path = 'Picture.h5'
try:
    file = h5py.File(path, 'r+')
    X_data = np.array(file["/dataset"]).astype("uint8")
    y_data = np.array(file["/label"]).astype("uint8")
    file.close()
except (FileNotFoundError, KeyError) as e:
    logging.error(f"Error loading data: {e}")
    exit()

y = np.ravel(y_data)

# ------------------- LDA feature Extraction ------------------------------------
lda = LDA(n_components=2)
lda.fit(X_data, y_data.ravel())

# ---------------------------------- TKINTER ---------------------------------------
def select_image():
    global mylabel_knn, mylabel_mlp, mylabel_svm, imgdisplay

    img_path = filedialog.askopenfilename()
    if not img_path:
        return
    
    img = cv.imread(img_path)
    if img is None:
        logging.error("Invalid image file")
        return

    img_resized = cv.resize(img, (400, 200))
    img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
    inputimg = img_gray.flatten().reshape(1, -1)

    inputimg = lda.transform(inputimg)

    probability_knn = model_knn.predict_proba(inputimg)
    probability_mlp = model_mlp.predict_proba(inputimg)
    probability_svm = model_svm.predict_proba(inputimg)
    Categories = ['BSN', 'Maybank', 'Public Bank']

    # -------loop for the categories information--------------
    label_knn = [f"{val} = {probability_knn[0][ind] * 100:.2f}%" for ind, val in enumerate(Categories)]
    label_mlp = [f"{val} = {probability_mlp[0][ind] * 100:.2f}%" for ind, val in enumerate(Categories)]
    label_svm = [f"{val} = {probability_svm[0][ind] * 100:.2f}%" for ind, val in enumerate(Categories)]

    logging.info("KNN: %s", label_knn)
    logging.info("MLP: %s", label_mlp)
    logging.info("SVM: %s", label_svm)

    final_test_knn = int(model_knn.predict(inputimg))
    final_test_mlp = int(model_mlp.predict(inputimg))
    final_test_svm = int(model_svm.predict(inputimg))

    test_categories = 'The Predicted Image is: '
    cat_knn = Categories[final_test_knn]
    cat_mlp = Categories[final_test_mlp]
    cat_svm = Categories[final_test_svm]

    cont_knn = test_categories + cat_knn + '\n\n'
    cont_mlp = test_categories + cat_mlp + '\n\n'
    cont_svm = test_categories + cat_svm + '\n\n'

    image = Image.open(img_path)
    image = image.resize((500, 200), Image.LANCZOS)
    imgdisplay = ImageTk.PhotoImage(image)

    Label(root, image=imgdisplay).grid(row=4, column=0)
    Label(root, text="KNN Prediction", font=("Helvetica", 14, "bold")).grid(row=5, column=0)
    Label(root, text="MLP Prediction", font=("Helvetica", 14, "bold")).grid(row=8, column=0)
    Label(root, text="SVM Prediction", font=("Helvetica", 14, "bold")).grid(row=11, column=0)

    Label(root, text="\n".join(map(str, label_knn)), font=("Helvetica", 12)).grid(row=6, column=0, sticky="S")
    Label(root, text="\n".join(map(str, label_mlp)), font=("Helvetica", 12)).grid(row=9, column=0, sticky="S")
    Label(root, text="\n".join(map(str, label_svm)), font=("Helvetica", 12)).grid(row=12, column=0, sticky="S")

    if mylabel_knn.winfo_exists():
        mylabel_knn.destroy()
    mylabel_knn = Label(root, text=cont_knn, font=("Helvetica", 12, "bold"))
    mylabel_knn.grid(row=7, column=0)

    if mylabel_mlp.winfo_exists():
        mylabel_mlp.destroy()
    mylabel_mlp = Label(root, text=cont_mlp, font=("Helvetica", 12, "bold"))
    mylabel_mlp.grid(row=10, column=0)

    if mylabel_svm.winfo_exists():
        mylabel_svm.destroy()
    mylabel_svm = Label(root, text=cont_svm, font=("Helvetica", 12, "bold"))
    mylabel_svm.grid(row=13, column=0)

    root.mainloop()

root = tk.Tk()
root.title("Image Classification")
root.geometry('600x1000')
paths = 'logo/logo.JPG'
try:
    load = Image.open(paths)
    render = ImageTk.PhotoImage(load)
    main_logo = load.resize((150,150), Image.LANCZOS)
    logodisplay = ImageTk.PhotoImage(main_logo)
    root.iconphoto(False, render)
except FileNotFoundError:
    logging.error("Logo file not found")

Label(root, image=logodisplay).grid(row=0, column=0, pady=5)

mylabel_svm = Label(root)
mylabel_mlp = Label(root)
mylabel_knn = Label(root)
Label(root, text="Bank Pattern Recognition", font=("Helvetica", 20, "bold")).grid(row=1, sticky=N, rowspan=2)
Button(root, text="Select Image", height=2, width=18, command=select_image).grid(row=3, column=0)

root.mainloop()

