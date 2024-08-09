import React, { useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Button } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';
import axios from 'axios';

// import { checkServer } from '../src/api';

const AboutScreen = ({navigation}) => {
  const checkServer = async () => {
    try {
        const response = await axios.get("http://192.168.1.49:5000/", {
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