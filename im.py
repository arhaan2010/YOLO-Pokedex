import streamlit as st
from ultralytics import YOLO
from PIL import Image
st.set_page_config(layout= "wide")

tab1, tab2, tab3 = st.tabs(["Pokedex", "About-YOLO", "Why?"])
col1, col2, col3 = st.columns([1, 2, 3])
with tab1:
    
    st.markdown("""
                <style>
                        @import url('https://fonts.googleapis.com/css2?family=Major+Mono+Display&display=swap"');
                        @import url('https://fonts.googleapis.com/css2?family=Teko:wght@300..700&display=swap" rel="stylesheet');
                    .stApp {
                         background-color:#F0E600;
                         background-size: cover;
                         background-repeat: no-repeat;
                         background-attachment: fixed;
                
                         }
                .title h1 {
                            
                           position: absolute;
                                top: 290px; /* Change top value for vertical position */
                                left: 80% ; /* Center horizontally */
                                transform: translateX(-50%); /* Adjust for center alignment */
                                color: #DF9D20;
                                font-family: "Major Mono Display", monospace;
                                font-style: normal;
                                 width: 100%;
                         }
                  .positioned-image {
                position: absolute;
                top: 0%; /* Change top value for vertical position */
                left: 15%; /* Change left value for horizontal position */
            }
            .body h1 {
                    color: #DF9D20;
                    font-family: "Teko",sans-serif;
                    font-style: normal;
                    }
              
        </style>
        <div class="positioned-image">
            <img src="https://media.discordapp.net/attachments/1093112144515579965/1245706197949218860/International_Pokemon_logo.svg.png?ex=665effcd&is=665dae4d&hm=51f0d736a89f97fbf83cdc85506da2e1946dcb42db43ff68cbdd39c3faa6a657&=&format=webp&quality=lossless&width=687&height=252" width="869">
        </div>
        <div class="pokemon1">
            <img src="" width="869">
        </div>
        <div class="pokemon2">
            <img src="" width="869">
        </div>
    """, unsafe_allow_html=True)    
    st.markdown('<div class="title"><h1>your own pokedex</h1></div>', unsafe_allow_html=True)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

# Add some spacing at the top of the site
st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

# Image uploader section
img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if img is not None:
    img = Image.open(img)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    model = load_model()
    results = model.predict(img)

    # Assuming the results contain probabilities for classes
    pred = results[0].probs.top1  # Access the top prediction

    # Display the predicted class name
    results = model.predict(img)
    st.write(f"Your Pokemon Is: {results[0].names[pred]}")
       # Format the search URL
    search_url = f"https://www.pokemon.com/us/pokedex/{results[0].names[pred]}"
        
        # Display the search link
    st.markdown(f"[Know more about {results[0].names[pred]}?]({search_url})")
        


with tab2:
    st.markdown("""
                <div class="body"><h1 style= "color:#299CD6;">YOLO (You Only Look Once) is a popular object detection algorithm that can detect multiple objects in an image and precisely localize them. It's widely used in various applications such as self-driving cars, security systems, and more.<br>
                1. Image Input
                The input to the YOLO algorithm is an image. The algorithm processes the entire image at once, unlike other object detection methods that scan parts of the image multiple times at different scales.
                <br>
                2. Grid Division
                The image is divided into an 
                S×S grid of cells. Each cell is responsible for detecting objects whose center falls within it. This grid-based approach helps in localizing objects in the image.
                <br>
                3. Bounding Box Prediction
                Each cell predicts a fixed number of bounding boxes (usually denoted by 
                B, e.g., 2 or 3 boxes). For each bounding box, the following attributes are predicted:
                x and y coordinates of the box center relative to the bounds of the grid cell
                Width and height of the box (relative to the entire image)
                Confidence score for the box (how likely it is that the box contains an object and how accurate the bounding box is)
                   <br>
                4. Class Prediction
                Each cell also predicts class probabilities for the object within the bounding box. This is a set of probabilities over all the classes, indicating what kind of object (e.g., car, person, dog) is in the box.
                <br>
                5. Confidence Score Calculation
                The confidence score for each predicted bounding box is calculated as:
                Confidence
               <br>
                6. Prediction Consolidation
                The model outputs a large number of bounding boxes, most of which may overlap significantly or may not contain objects at all. YOLO uses non-max suppression to eliminate redundant boxes. This step involves:
                <br>
                Discarding boxes with low confidence scores.
                Selecting the box with the highest confidence score and suppressing all other boxes that overlap significantly with it.
                7. Final Output
                The final output is a list of bounding boxes with their associated class labels and confidence scores, providing the locations and types of objects detected in the image.
                <br>
                Summary
                YOLO's approach can be summarized as:
                
                Single-pass detection: Unlike other methods that use region proposals and apply the classifier to each region, YOLO performs detection in a single forward pass of the network.
                Grid-based detection: The image is divided into a grid, and predictions are made on a per-cell basis.
                Unified architecture: YOLO is a single convolutional network that simultaneously predicts multiple bounding boxes and class probabilities for those boxes.</h1></div>

                """, unsafe_allow_html=True)
    st.image("yolo.png")



with tab3:
    st.markdown("""
              <div class="body">  <h1 style= "color:#299CD6 ;">As a child, the allure of Pokémon captivated many of us, inspiring dreams of adventure, camaraderie, and the quest to "catch 'em all." Among the many treasures within the Pokémon universe, the Pokédex stood out as an emblem of knowledge and exploration, serving as a vital companion on the journey to becoming a Pokémon Master. For those of us who grew up watching Pokémon, the desire for a real-life Pokédex was not merely a whimsical fantasy but a heartfelt yearning rooted in our deep connection to the franchise and its world.</h1></div>
                """, unsafe_allow_html=True)
