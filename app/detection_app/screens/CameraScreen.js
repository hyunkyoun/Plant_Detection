import React from 'react'
import { CameraView, useCameraPermissions } from 'expo-camera';
import { shareAsync } from 'expo-sharing';
import * as MediaLibrary from 'expo-media-library';
import { useState, useRef, useEffect } from 'react';
import { StyleSheet, Text, View, Button, TouchableOpacity, ImageBackground } from 'react-native';
import axios from 'axios';
import FormData from 'form-data'

import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import Entypo from '@expo/vector-icons/Entypo';
import Ionicons from '@expo/vector-icons/Ionicons';

const CameraScreen = ({ navigation }) => {
  const [facing, setFacing] = useState('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [image, setImage] = useState(null);
  const cameraRef = useRef();
  const [response, setResponse] = useState(null);

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View style={styles.permission}>
        <Text style={{ textAlign: 'center' }}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  // function to switch the camera from front to back facing one
  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync();
        setImage(photo.uri);
      } catch(e) {
        console.log(e);
      }
    }
  }

  const returnToCamera = async () => {
    setImage(null);
  }

  const sendToModel = async () => {

    const formData = new FormData();
    formData.append('file', {
      uri: image,
      name: 'photo.jpg',
      type: 'image/jpeg',
    });

    try {
        const response = await axios.post("http://192.168.1.49:5000/api/sendToModel", formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        console.log('Response: ', response.data);
    } catch(error) {
        console.error('Error connecting to server', error);
    };
  };

  return (
    <View style={styles.container}>
      {!image ? 
      // Camera View before taking picture
        <CameraView style={styles.camera} type={facing} ref={cameraRef}>
          <View style={styles.info_button_container}>
            {/* Information button */}
            <TouchableOpacity style={styles.infoButton} onPress={() => navigation.navigate('Info')}>
              <Ionicons name="information-circle-outline" size={30} color="white" />
            </TouchableOpacity>
          </View>
          <View style={styles.buttonContainer}>
            <View style={styles.camera_button_container}>
              {/* Take picture button */}
              <TouchableOpacity style={styles.shutter} onPress={takePicture}>
                <Entypo name="picasa" size={50} color="white" />
              </TouchableOpacity>

              {/* Flip camera button */}
              <TouchableOpacity style={styles.flipButton} onPress={toggleCameraFacing}>
                <FontAwesome6 name="camera-rotate" size={25} color="white" />
              </TouchableOpacity>
            </View>
          </View>
        </CameraView>
      :
      // View after taking picture
        <View style={styles.container}>
          <ImageBackground source={{uri: image}} style={styles.camera}>
            <View style={styles.buttonContainer}>
              <View style={styles.camera_button_container}>
                {/* Send to model */}
                <TouchableOpacity style={styles.postPictureCheck} onPress={sendToModel}>
                  <Entypo name="check" size={50} color="white" />
                </TouchableOpacity>
                {/* Return to camera button */}
                <TouchableOpacity style={styles.postPictureX} onPress={returnToCamera}>
                  <Entypo name="cross" size={50} color="white" />
                </TouchableOpacity>
              </View>
            </View>
          </ImageBackground>
        </View>
      }
      
    </View>
  );
}

export default CameraScreen;

const styles = StyleSheet.create({
  permission: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },  
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  camera: {
    flex: 1,
  },
  info_button_container: {
    flex: 1,
    flexDirection: 'row',
    margin: 25,
  },
  infoButton: {
    left: '85%',
    top: 35,
  },
  buttonContainer: {
    flex: 1,
    margin: 30,
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  flipButton: {
    position: 'absolute',
    right: 10,
  },
  camera_button_container: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 50,
    width: '100%',
    justifyContent: 'center',
    bottom: 10,
  },
  postPictureCheck: {
    right: 50,
  },
  postPictureX: {
    left: 50,
  },
});

