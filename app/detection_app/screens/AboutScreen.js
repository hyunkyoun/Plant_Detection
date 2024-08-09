import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Button } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';
import axios from 'axios';

const AboutScreen = ({navigation}) => {

  const [inputData, setInputData] = useState('');
  const [responseData, setResponseData] = useState(null);

  // setInputData("input data");

  const sendData = () => {
    try {
      const res = axios.post('http://localhost:5001/api/process', { key1: "value1", key2: "value2" })
        .then((res) => {
          console.log(res);
          setResponseData(res.data);
        })
        .catch((error) => {
          console.error(error)
        })
    } catch (err) {
      console.error(err);
    }
  }

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
      </View>
      <View style={styles.button_container}>
        <Button 
          onPress={sendData}
          title="Press"
        />
        <Text>Response: </Text>
        <Text>{JSON.stringify(responseData)}</Text>
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
  button_container: {
    flex: 1, 
  }, 
});

export default AboutScreen;