import React, { useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Button } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';
import axios from 'axios';

const AboutScreen = ({navigation}) => {
  const checkServer = async () => {
    const IP_ADDRESS = 'http://192.168.1.49';
    const PORT = '5001';

    try {
        const response = await axios.get(`${IP_ADDRESS}:${PORT}`, {
          timeout: 5000,
        });
        console.log('Response: ', response.data);
    } catch(error) {
        console.error('Error connecting to server', error);
    }
  }
  
  useEffect(() => {
    checkServer();
  }, []);

  return (
    <View style={styles.container}>
      <View style={styles.header}>

        {/* back button to go to the home / camera page */}
        <TouchableOpacity onPress={() => navigation.navigate('Home')}>
          <AntDesign name="left" size={30} color="black" />
        </TouchableOpacity>

      </View>

      {/* text in the about screen */}
      <View style={styles.text_container}>
        <Text>This is the about screen!</Text>
        <Button title="Check Server" onPress={checkServer} />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    height: 100,
    justifyContent: 'flex-end',
    margin: 10,
  },
  text_container: {
    flex: 1,
    alignItems: 'center',
  },
});

export default AboutScreen;