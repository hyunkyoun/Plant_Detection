import React, { useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';

const AboutScreen = ({navigation}) => {
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