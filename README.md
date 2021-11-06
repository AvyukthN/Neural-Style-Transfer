# Neural Style Transfer Algorithm

Simple NST algorithm implemented in TensorFlow according to the original paper "A Neural Algorithm of Artisic Style" - https://arxiv.org/abs/1508.06576

NST is a deep learning method that takes the style of one image and applies it to the content in another image. The style and content are captured in the convolutional layers of a pretrained VGG-19 as described in the original paper. The algorithm implemented works by taking the content image and slowly changing it's pixel values according to the style.

## Implementation Details
**Total Loss**: A total loss was calculated using the NST loss function (alpha * content_loss  + beta * style_loss).

**Content Loss**: Content loss was found by calculating the element-wise difference (also known as the "Euclidian Norm") in activations between the content and generated image from the VGG-19 model

**Style Loss**: Style loss is measured by the correlation between activations across different channels in some layer l in the VGG-19 model. A gram matrix is calculated for both the style and generated image (in this case the image that we are applying style to), by finding the "un-normalized cross-covariance" which is simply the sum of the product of activations across pixels i and channels j. Finally the Euclidian norm is found for the two gram matrices which gives us the style loss.

**Completing the Model**: The content and style loss are combined to find the total loss as shown in the "Total Loss" section above. The total loss is then used to calculate gradients, later optimized using an adam optimizer, which are applied to the content image to generate our final combined image.