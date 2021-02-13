<h1 align="center">:hospital: X Ray Anonymiser :hospital:</h1>


<div align="center">
<img src="Assets\MIT-Private-Browsing_0.jpg" width=672px height=450px/> 
<br>
</div>



[![](https://img.shields.io/badge/Made_with-Python-green?style=for-the-badge&logo=python)](https://www.python.org)
[![](https://img.shields.io/badge/Made_with-nltk-green?style=for-the-badge&logo=nltk)](https://www.nltk.org/#)
[![](https://img.shields.io/badge/Made_with-tesseract-green?style=for-the-badge&logo=tesseract)](https://opensource.google/projects/tesseract)
[![](https://img.shields.io/badge/Made_with-Opencv-green?style=for-the-badge&logo=opencv)](https://opencv.org)

<br>

</div>

---


<h2>Aim:</h2>

Healthcare industry has seen a massive influx of data from a plethora of digital sources. This data has opened the doors to many new opportunities in providing timely and accurate treatments to patients and access to information that can be used for scientific research. While this data does provide an abundance of valuable insights, one of the major challenges in healthcare data is the privacy, security, and confidentiality of the patient's data. We attempt to tackle one of the aspect of securing the privacy of the data i.e. anonymizing names present in a particular health record namely X-rays.

<h2>Objectives:</h2>

1) To create a program capable of taking image(s) as input, identifying, localising and classifying the text, filtering out the proper nouns, and black out (anonymize) the filtered words present.

2) To provide additional functionalities like flip, rotate, undo and saving of the image.


<h2>GUI:</h2>
The GUI is designed using matplotlib integrated with tkinter module. The image window is a matplotlib plot, whereas the buttons are made using tkinter.

<div align="center">
<img src="Assets\GUI.png" width=672px height=450px/> 
<br>
Figure 1: Tkinter GUI window
<br>
</div>

<h2>Procedure</h2>
<h3>Pre-processing of the input image</h3>
Every image is converted to <b>grayscale</b> for ease and increased speed of computation.
<h3>Blur Checking</h3>
Blur images are of two types- 1)Motion blur and 2)Focus blur
Every image is <b>resized</b> to a fixed size (by height) <b>maintaining the aspect ratio</b> after which a Laplacian kernel is convolved over it. The variance is further calculated.
If the variance is less than a predetermined threshold (<b>500 units</b>), the image is considered to be blurry.

<div align="center">
<img src="Assets\Blur_Correction.png" width=672px height=450px/> 
<br>
Figure 2: Laplacian Variance of different image samples
<br>
</div>
<h3>Blur Correction</h3>
Blur images have lower intensity pixels surrounding higher intensity pixels
So a <b>threshold</b> (binary inverse) is applied on the images, with a threshold pixel value of 140. This causes all pixel values below 140 to be converted to 255, and all values above 140 to be converted to 0. This improves the clarity of the text since the lower intensity pixels
are thresholded.
<div align="center">
<img src="Assets\Blur_Thresh.png" width=672px height=300px/> 
<br>
Figure 3: Effect of blur correction
<br>
</div>
<h3>Inversion of Images</h3>
Tesseract 4.x works efficiently for <b>dark text on light background</b>. Since most of the x-rays had white text on black background, inversion of the images was necessary. A global inversion could not be directly applied since images could be different in nature. So, a histogram of the pixel intensities was taken and the variance was calculated. If more pixels have intensities towards 0, this indicates the image is mainly black and subsequently the text would be white, which would require an inversion. The inversion is done by simply applying a bitwise not function on the image.

<h2>Text recognition, location and classification</h2>
<h3>Tesseract OCR</h3>
OCR stands for "Optical Character Recognition". OCR systems transform a two-dimensional image of text, that could contain machine printed or handwritten text from its image representation into machine-readable text. OCR as a process generally consists of several sub-processes to perform as accurately as possible, which includes pre-processing of the image, text localization, character segmentation, character recognition and post processing.

In this program, we opted for the Tesseract OCR engine (namely Tesseract 4.0) which is an open source text recognition engine.

Tesseract 3.x is based on traditional computer vision algorithms. However, in the past few years, Deep Learning based methods have surpassed traditional machine learning techniques by a huge margin in terms of accuracy in many areas of Computer Vision. Handwriting recognition is one of the prominent examples.  So, in version 4, Tesseract has implemented a Long Short Term Memory (<b>LSTM</b>) based recognition engine. LSTM is a kind of Recurrent Neural Network (RNN).

Processing of text follows a traditional step-by-step pipeline. The first step is a connected component analysis in which outlines of the components are stored. Then, the outlines are gathered together, purely by nesting, into Blobs. Blobs are organized into text lines, and the lines and regions are analysed for fixed pitch or proportional text. Text lines are broken into words differently according to the kind of character spacing. Fixed pitch text is chopped immediately by character cells. Proportional text is broken into words using definite spaces and fuzzy spaces.

Recognition then proceeds as a two-pass process. In the first pass, an attempt is made to recognize each word in turn. Each word that is satisfactory is passed to an adaptive classifier as training data. The adaptive classifier then gets a chance to more accurately recognize text lower down the page. Since the adaptive classifier may have learned something useful too late to make a contribution near the top of the page, a second pass is run over the page, in which words that were not recognized well enough are recognized again.
A final phase resolves fuzzy spaces, and checks alternative hypotheses for the x-height to locate small-cap text.

Thus, the working of tesseract can be summarized in 5 steps:
    Line and word finding
    Word recognition
    Static character classification
    Adaptive classification
    Linguistic analysis
   
So, in our program, after the image is pre-processed, it is passed to the OCR function (which is invoked when the user clicks the "Anonymize" button on the GUI), which utilizes the tesseract module.
Tesseract then works on the image (in the above specified steps) and gives a string output of the text it found.

<div align="center">
<img src="Assets\Tesseract_Output.png" width=672px height=450px/> 
<br>
Figure 4: Tesseract Output
<br>
</div>

<h3>Natural Language Processing (NLP)</h3>
The output of tesseract is then passed to the NLP function which utilizes the <b>"Natural Language Toolkit"</b> i.e. the nltk module.

Steps for Named Entity Extraction:
1.	Tokenization: The Input text is tokenized based on pre trained NLP models from the Natural Language Toolkit Module.

2.	Classification: Since the given text doesn’t form proper English sentence (Because the input text is medical data), standard classification procedure wasn’t applicable. We determined that a given word is a Named Entity if it’s a <b>proper noun (NNP)</b> and is followed by a <b>proper noun (NNP)</b> or a <b>common noun(NN)</b>.

<h3>Fuzzy Matching</h3>
The output from the NLP consists of words that are proper nouns. However, due to noise in the image resulting in undesirable words in the tesseract output as well as the NLP output, it was necessary to add another layer of filtering. That's where fuzzy matching comes in.

Fuzzy matching is a concept based on the <b>Levenshtein</b> distance (also called as edit distance), which is basically the amount of substitutions, insertions or deletions required for one word to be transformed to another. So shorter the levenshtein distance, more similar are the words.

<b>"fuzzywuzzy"</b> is a module which utilizes the Levenshtein distance. In this program, couple of functions of the module are used, namely <b>"ratio"</b> and <b>"extractOne"</b>. The ratio function returns the percentage of the similarity (ranging from 0 to 100) of two words which are passed in as parameters, whereas the extractOne function compares a given string with a list of strings and returns a tuple of the highest matched string from that list along with the matched percentage.

In this way, the output text from the NLP function is "fuzzy matched" with 2 datasets of common Indian first names (approximately 6000) and last names (approximately 1500).
If the matched percentage is greater than a predetermined value (<b>here 60</b>), the word is considered to be a proper noun and is included in the final output where it is anonymized.
The accuracy and speed of this step depends on the dataset used.

<h2>Additional Functionalities</h2>
The program also includes functionalities to flip the image (horizontally and vertically), rotate the image (clockwise and anticlockwise), cropping (auto and manual, both explained below), undo (which utilizes stack data structure) and save processed image.


<h3>Cropping</h3>
<h4>Auto Crop</h4>
This functionality is used to extract the desired image from unwanted background.

Steps for achieving auto crop:

a. The image is <b>resized</b> while maintaining its aspect ratio for the kernels (filters) to work effectively. 

b. <b>Median blur</b> is applied to the cropped image for noise reduction.

c. <b>Thresholding</b> (Adaptive Threshold) and <b>Closing</b> are applied to get a thick boundary around the document (Assuming contrasting background). <b>Canny</b> was applied to the thresholded image to get 1 pixel thick edges (borders).

d. <b>Contours</b> are applied on the canny image and their shape was approximated. If the approximated shape had 4 points (vertices), then it was warped, giving us the desired document.

<h4>Manual Crop</h4>
A mouse callback function is used to retrieve 4 points the user selects on the image. These points are appended in a global array and are used to perform a <b>warp perspective transform</b>.

<h2>Results</h2>
A mouse callback function is used to retrieve 4 points the user selects on the image. These points are appended in a global array and are used to perform a warp perspective transform.

<div align="center">
<img src="Assets\Res_1.png" width=672px height=300px/> <br>
Figure 5: Image sample with motion blur
<br>
</div>


<div align="center">
<img src="Assets\Res_1.png" width=672px height=300px/> 
<br>
Figure 6: Image sample with focus blur
<br>
</div>


<div align="center">
<img src="Assets\Res_1.png" width=672px height=300px/> 
<br>
Figure 7: Image sample with <br>handwritten text

</div>


### Conclusion and Futher Work:
  * [x] The application program is able to handle computerised text as well as handwritten text for processing.
  * [x] It is able to handle different kinds of blurs, namely motion blur and focus blur caused due to human error making it robust enough for commercial application.
  * [x] The program can be extended to include additional functionalities like identifying and anonymizing phone number and other personal information.

<h2>References</h2>
<br>

1.	http://www.ijstr.org/final-print/nov2019/The-Importance-Of-Preserving-The-Anonymity-In-Healthcare-Data-A-Survey-.pdf
2.	https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33418.pdf
3.	https://nanonets.com/blog/ocr-with-tesseract/
4.	https://towardsdatascience.com/string-matching-with-fuzzywuzzy-e982c61f8a84
5.	https://likegeeks.com/nlp-tutorial-using-python-nltk/
<br>

---

<h3 align="center"><b>Developed by Anish Pawar and Chaitanya Bandiwdekar</b></h3>

[![](https://img.shields.io/badge/LinkedIn-Anish_Pawar-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/anish-pawar-5300a9192/)

[![](https://img.shields.io/badge/LinkedIn-Chaitanya_Bandiwdekar-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/chaitanya-bandiwdekar-11329a18a/)