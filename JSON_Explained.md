# The annotation file will be of COCO format and have the following structure

# images
## id: Image id
## width: Width of the image
## height: Height of the image
## filename: Image file name
## license: License id for the image
## date_captured: Date of capture of the image


# annotations
## id: Annotation id
## image_id: Id of the image the annotation is associated with
## category_id: Id of the class the annotation belongs to
## segmentation: (x, y) coordinates of the four corners of the bounding box
## area: Area of the bounding box
## bbox: (x, y) coordinate of the top-left corner and width and height of the bounding box
## iscrowd: If the image has a crowd of objects denoted by this annotation





# Following is documentation taken and explained from the newest tool

{
  "project": {                       
  # ["project"] contains all metadata associated with this VIA project
    "pid": "__VIA_PROJECT_ID__",     
    # uniquely identifies a shared project (DO NOT CHANGE)
    "rev": "__VIA_PROJECT_REV_ID__", 
    # project version number starting form 1 (DO NOT CHANGE)
    "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__", # commit timestamp for last revision (DO NOT CHANGE)
    "pname": "VIA3 Sample Project",  
    # Descriptive name of VIA project (shown in top left corner of VIA application)
    "creator": "VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via)",
    "created": 1588343615019,        
    # timestamp recording the creation date/time of this project (not important)
    "vid_list": ["1", "2"]           
    # selects the views that are visible to the user for manual annotation (see ["view"])
  },
  "config": {           
  # Configurations and user settings (used to modify behaviour and apperance of VIA application)
    "file": {
      "loc_prefix": {   
      # a prefix automatically appended to each file 'src' attribute. Leave it blank if you don't understand it. See https://gitlab.com/vgg/via/-/blob/master/via-3.x.y/src/js/_via_file.js
        "1": "",        
        # appended to files added using browser's file selector (NOT USED YET)
        "2": "",        
        # appended to remote files (e.g. http://images.google.com/...)
        "3": "",        
        # appended to local files  (e.g. /home/tlm/data/videos)
        "4": ""         
        # appended to files added as inline (NOT USED YET)
      }
    },
    "ui": {
      "file_content_align": "center",
      "file_metadata_editor_visible": true,
      "spatial_metadata_editor_visible": true,
      "spatial_region_label_attribute_id": ""
    }
  },
  "attribute": {       
  # defines the things that a human annotator will describe and define for images, audio and video.
    "1": {                          
    # attribute-id (unique)
      "aname":"Activity",           
      # attribute name (shown to the user)
      "anchor_id":"FILE1_Z2_XY0",   
      # FILE1_Z2_XY0 denotes that this attribute define a temporal segment of a video file. See https://gitlab.com/vgg/via/-/blob/master/via-3.x.y/src/js/_via_attribute.js
      "type":4,                     
      # attributes's user input type ('TEXT':1, 'CHECKBOX':2, 'RADIO':3, 'SELECT':4, 'IMAGE':5 )
      "desc":"Activity",            
      # (NOT USED YET)
      "options":{"1":"Break Egg", "2":"Pour Liquid", "3":"Cut Garlic", "4":"Mix"}, # defines KEY:VALUE pairs and VALUE is shown as options of dropdown menu or radio button list
      "default_option_id":""
    },
    "2": {
      "aname":"Object",
      "anchor_id":"FILE1_Z1_XY1",   
      # FILE1_Z1_XY1 denotes attribute of a spatial region (e.g. rectangular region) in a video frame. See https://gitlab.com/vgg/via/-/blob/master/via-3.x.y/src/js/_via_attribute.js
      "type":1,                     
      # an attribute with text input
      "desc":"Name of Object",
      "options":{},                 
      # no options required as it has a text input
      "default_option_id":""
    }
  },
  "file": {                         
  # define the files (image, audio, video) used in this project
    "1":{                           
    # unique file identifier
      "fid":1,                      
      # unique file identifier (same as above)
      "fname":"Alioli.ogv",         
      # file name (shown to the user, no other use)
      "type":4,                     
      # file type { IMAGE:2, VIDEO:4, AUDIO:8 }
      "loc":2,                      
      # file location { LOCAL:1, URIHTTP:2, URIFILE:3, INLINE:4 }
      "src":"https://upload.wikimedia.org/wikipedia/commons/4/47/Alioli.ogv" # file content is fetched from this location (VERY IMPORTANT)
    },
    "2":{
      "fid":2,
      "fname":"mouse.mp4",
      "type":4,
      "loc":3,                      # a file residing in local disk (i.e. file system)
      "src":"/home/tlm/data/svt/mouse.mp4"
    }
  },
  "view": {           
  # defines views, users see the "view" and not file, each view contains a set of files that is shown to the user
    "1": {            
    # unique view identifier
      "fid_list":[1]  
      # this view shows a single file with file-id of 1 (which is the Alioli.ogv video file)
    },
    "2": {
      "fid_list":[2]  
      # this view also shows a single video file (but a view can contain more than 1 files)
    }
  },
  "metadata": {       
  # a set of all metadata define using the VIA application
    "-glfwaaX": {     
    # a unique metadata identifier
      "vid": "1",     
      # view to which this metadata is attached to
      "flg": 0,       
      # NOT USED YET
      "z": [2, 6.5],  
      # z defines temporal location in audio or video, here it records a temporal segment from 2 sec. to 6.5 sec.
      "xy": [],       
      # xy defines spatial location (e.g. bounding box), here it is empty
      "av": {         
      # defines the value of each attribute for this (z, xy) combination
        "1":"1"       
        # the value for attribute-id="1" is one of its option with id "1" (i.e. Activity = Break Egg)
      }
    },
    "+mHHT-tg": {
      "vid": "1",
      "flg": 0,
      "z": [9, 20],
      "xy": [],
      "av": {
        "1":"2"      
        # the value for attribute-id="1" is one of its option with id "2" (i.e. Activity = Pour Liquid)
      }
    },
    "ed+wsOZZ": {
      "vid": "1",
      "flg": 0,
      "z": [24, 26],
      "xy": [],
      "av": {
        "1":"2"
      }
    },
    "fH-oMre1": {
      "vid": "1",
      "flg": 0,
      "z": [0.917],                
      # defines the video frame at 0.917 sec.
      "xy": [4, 263, 184, 17, 13], 
      # defines a rectangular region at (263,184) of size (17,13). The first number "4" denotes a rectangular region. Other possible region shapes are: { 'POINT':1, 'RECTANGLE':2, 'CIRCLE':3, 'ELLIPSE':4, 'LINE':5, 'POLYLINE':6, 'POLYGON':7, 'EXTREME_RECTANGLE': 8, 'EXTREME_CIRCLE':9 }
      "av": {
        "2":"Egg"                  
        # the value of attribute-id "2" is "Egg" (i.e. Object = Egg)
      }
    }
  }
}
