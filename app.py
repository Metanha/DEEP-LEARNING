# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Fonction de prétraitement générique
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    if image.ndim == 2:  # grayscale
        image = np.stack([image]*3, axis=-1)
    return np.expand_dims(image, axis=0)

# Chargement des modèles (à adapter avec tes chemins réels)
model_paludisme = tf.keras.models.load_model("Modele/modele_paludisme.h5")
#model_chien_chat = tf.keras.models.load_model("Modele/modele_dog93.h5")
model_cifar10 = tf.keras.models.load_model("Modele/modele_multiclasse74.h5")

# Classes CIFAR-10
classes_cifar10 = ['Avion', 'Voiture', 'Oiseau', 'Chat', 'Cerf', 'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion']

def interface_paludisme():
    st.header("🦠 Diagnostic Paludisme")
    image = st.file_uploader("Uploader une image de cellule", type=["jpg", "png"])
    if image:
        img = Image.open(image)
        st.image(img, caption="Image fournie", use_container_width=True)
        img_array=preprocess_image(img, (50,50))
        #st.write("Forme attendue par le modèle :", model_paludisme.input_shape)
        #st.write("Forme réelle de l’image :", img_array.shape)

        img_array = np.array(img_array) / 255.0
        pred = model_paludisme.predict(img_array)[0][0]
        result = "Infectée" if pred > 0.5 else "Non infectée"
        st.success(f"Résultat : {result} (probabilité = {pred:.2f})")

def interface_chien_chat():
    st.header("🐶🐱 Classification Chien vs Chat")
    image = st.file_uploader("Uploader une image d'animal", type=["jpg", "png"])
    if image:
        img = Image.open(image)
        st.image(img, caption="Image fournie", use_container_width=True)
        img = preprocess_image(img, (128,128))
        img_array = np.array(img) / 255.0
        st.write("Forme attendue par le modèle :",model_chien_chat.input_shape)
        st.write("Forme réelle de l’image :", img.shape)

        pred = model_chien_chat.predict(img_array)[0][0]
        result = "Chat" if pred > 0.5 else "Chien"
        st.success(f"Résultat : {result} (probabilité = {pred:.2f})")

def interface_cifar10():
    st.header("🎨 Classification CIFAR-10 (10 classes)")
    image = st.file_uploader("Uploader une image (32x32 minimum)", type=["jpg", "png"])
    if image:
        img = Image.open(image)
        #st.image(img, caption="Image fournie", use_container_width=True, width=15)
                # Prétraitement + reshaping correct
        img = preprocess_image(img, (71,71))
        img_array = np.array(img) / 255.0

        pred = model_cifar10.predict(img_array)[0]
        class_idx = np.argmax(pred)
        confiance=pred[class_idx]

        # Disposition en 2 colonnes
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(img, caption="Image", width=150)

        with col2:
            st.subheader("🧠 Résultat de la prédiction")
            st.markdown(f"**Classe prédite :** {classes_cifar10[class_idx]}")
            st.markdown(f"**Confiance :** {confiance:.2%}")
            st.progress(float(confiance))

        #st.success(pred)
        #st.success(f"Classe prédite : {classes_cifar10[class_idx]} (probabilité = {pred[class_idx]:.2f})")

# Page d'accueil
st.title("🧠 Application Deep Learning")
st.markdown("Bienvenue dans l'application de démonstration Deep Learning.")
st.markdown("Choisissez un modèle à utiliser dans le menu de gauche.")

# Interface principale via menu
choix = st.sidebar.radio("🔍 Choisir un modèle à explorer :", ("Accueil", "Paludisme", "Chien/Chat", "CIFAR-10"))

if choix == "Paludisme":
    interface_paludisme()
elif choix == "Chien/Chat":
    interface_chien_chat()
elif choix == "CIFAR-10":
    interface_cifar10()
else:
    st.image("images/dl.png", caption="Deep Learning - Classification d'Images", width=350)
    #st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Artificial_Intelligence_logo.svg/512px-Artificial_Intelligence_logo.svg.png", width=300)
    st.info("Sélectionnez un modèle dans le menu pour commencer.")

