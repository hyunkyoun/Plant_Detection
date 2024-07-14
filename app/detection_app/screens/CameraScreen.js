import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import Entypo from '@expo/vector-icons/Entypo';
import Ionicons from '@expo/vector-icons/Ionicons';


export default function App({navigation}) {
  const [facing, setFacing] = useState('back');
  const [permission, requestPermission] = useCameraPermissions();

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }

  // function to switch the camera from front to back facing one
  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing}>
        <View style={styles.info_button_container}>

          {/* information button */}
          <TouchableOpacity 
              style={styles.infoButton} 
              onPress={() => navigation.navigate('Info')}>
            <Ionicons name="information-circle-outline" size={30} color="white" />
          </TouchableOpacity>

        </View>
        <View style={styles.buttonContainer}>
          <View style={styles.camera_button_container}>

            {/* take picture button */}
            <TouchableOpacity style={styles.shutter}>
              <Entypo name="picasa" size={50} color="white" />
            </TouchableOpacity>

            {/* flip camera button */}
            <TouchableOpacity style={styles.flipButton} onPress={toggleCameraFacing}>
              <FontAwesome6 name="camera-rotate" size={25} color="white" />
            </TouchableOpacity>

          </View>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
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
  }
});

