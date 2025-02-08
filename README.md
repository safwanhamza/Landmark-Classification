# Landmark Classification & Tagging for Social Media

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

In this project, first steps have been taken towards addressing this problem by building models to automatically predict the location of the image based on any landmarks depicted in the image.

## Project Steps
The high level steps of the project include:

- Create a CNN to Classify Landmarks (from Scratch) - After visualizing the dataset, process it for training, and then build a convolutional neural network from scratch to classify the landmarks.

- Create a CNN to Classify Landmarks (using Transfer Learning) - Investigated different pre-trained models and decided one to use for this classification task.
  
- Write my own Landmark Prediction Algorithm - Used best model to create a simple interface for others to be able to use model to find the most likely landmarks depicted in an image.

## Review
To explore more about CNN architectures:
* https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96
* https://towardsdatascience.com/architecture-comparison-of-alexnet-vggnet-resnet-inception-densenet-beb8b116866d
