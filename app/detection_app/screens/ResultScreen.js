import React, { useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Button } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';

const ResultScreen = ({navigation, route}) => {

    const { species, confidence } = route.params;

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
        <Text>this flower is a {species}</Text>
        <Text>  confidence: {confidence}</Text>
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

export default ResultScreen;