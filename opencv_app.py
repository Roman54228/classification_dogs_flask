import cv2
import numpy as np 
import requests


class VideoAnalyser(object):
    r""" Video Analysis class 
    Args:
        video_path (string, required): path to input video

    """
    def __init__(self, video_path,downsample=8, wait=5, vis=True, write_vid=False):
        self.video_path = video_path
        self.setup_video_reader(self.video_path)
        self.downsample = downsample
        self.wait = wait
        self.vis = vis
        self.write_vid = write_vid
        if self.write_vid:
            self.setup_video_writer()

    def setup_video_reader(self, video_path):
        # setup video input   
        self.cap = cv2.VideoCapture(video_path)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"Input video stats: frame count {frame_count},",
                f"width {self.frame_width}, height {self.frame_height}, FPS: {fps}")
    
    def setup_video_writer(self, fn='out.mp4', fps=20.0):
        # fourcc =  cv2.cv.CV_FOURCC(*'MJPG')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
       
        self.video_writer = cv2.VideoWriter(fn, 
                                fourcc,fps, 
                                (self.frame_width, self.frame_height))
    
    def display_horiz_stacks(self,orig, filtered):
        """displays two frames stacked horizontally
        """
        cv2.imshow("display", np.concatenate([orig, filtered], axis=1))
        cv2.waitKey(self.wait)
    
    def apply_filter(self, frame):
        """applies filter or steps of filterring on the given input frame 
        """
        # fast image denoising on colored image 
        # converts to CIELAB format and applies 
        # denoising on L and ab components seperately
        dst = cv2.fastNlMeansDenoisingColored(frame.copy(),
                                    None, templateWindowSize=7, 
                                    searchWindowSize=21,
                                    h=3, hColor=3)
        return dst 

    def get_predictions(self, frame):
        resp = requests.post("http://localhost:5000/predict",
                     files={"file": cv2.imencode('.jpg', frame)[1].tobytes()})
        return resp.json()

    def plot_results(self, frame, results, alpha=0.25):
        overlay = frame.copy()
        xmin = 0
        ymin = self.frame_height//self.downsample - (self.frame_height//self.downsample)//8
        xmax = self.frame_width//self.downsample
        ymax = self.frame_height//self.downsample

        # xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax),
                    color=(255, 0, 0),thickness=-1)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha,0, frame)

        class_name = results['class_name']
        confidence = results['confidence']
        x = (self.frame_width//self.downsample)//4 + (self.frame_width//self.downsample)//16
        y =self.frame_height//self.downsample - (self.frame_height//self.downsample)//16

        cv2.putText(frame, 
                "Detected: {}, Score: {:.3f}".format(class_name,confidence), 
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        return frame


    def run(self):
        """ Runs the main loop
        """
        ret, frame = self.cap.read()
        while(ret):
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, 
                        (self.frame_width//self.downsample, self.frame_height//self.downsample))
            
            #filtered = self.apply_filter(frame)
            results = self.get_predictions(frame)
            
            if self.vis:
                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors, thickness=2)
                frame = self.plot_results(frame, results)
                cv2.imshow("display", frame)
                # self.display_horiz_stacks(frame, filtered)

                if cv2.waitKey(3) & 0xFF == ord('q'):
                    break
            if self.write_vid:
                self.video_writer.write(filtered)
        self.release_all()

    def release_all(self):
        # resurce release
        self.cap.release()
        if self.write_vid:
            self.video_writer.release()
        if self.vis:
            cv2.destroyAllWindows()   

# video_path='/Users/abhi/Downloads/IMG_0190.TRIM.MOV'
# downsample = 1
# video_analyser = VideoAnalyser(video_path=video_path, downsample=downsample, vis=False,write_vid=True )
# video_analyser.run()

def main():
    analyzer = VideoAnalyser(video_path=0,
                             downsample=2,
                             wait=3,
                             vis=True,
                             write_vid=False)
    analyzer.run()

if __name__ == "__main__":
    main()