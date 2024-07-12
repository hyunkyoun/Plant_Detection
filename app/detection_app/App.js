import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState } from 'react';
import { Button, StyleSheet, Text, Touchable, TouchableOpacity, View } from 'react-native';
import FontAwesome6 from '@expo/vector-icons/FontAwesome6';
import Entypo from '@expo/vector-icons/Entypo';


export default function App() {
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

  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing}>
        <View style={styles.buttonContainer}>
          <View style={styles.spacer}>
            <TouchableOpacity style={styles.shutter}>
              <Entypo name="picasa" size={50} color="white" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.flipButton} onPress={toggleCameraFacing}>
              <FontAwesome6 name="camera-rotate" size={30} color="white" />
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
  buttonContainer: {
    flex: 1,
    margin: 30,
    backgroundColor: 'blue',
    alignItems: 'center',
    justifyContent: 'flex-end'
  },
  flipButton: {
    // alignContent: 'flex-end',
    // alignItems: 'center',
    // alignSelf: 'flex-end',
  },
  shutter: {
    // alignItems: 'center',
    justifyContent: 'center'
  },
  spacer: {
    flexDirection: 'row',
    alignItems: 'center',
  }
});