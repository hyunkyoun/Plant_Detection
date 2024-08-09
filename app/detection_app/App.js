import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import AboutScreen from './screens/AboutScreen';
import CameraScreen from './screens/CameraScreen';
import ResultScreen from './screens/ResultScreen';

// this handles the navigation of the app

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen name="Home" component={CameraScreen} options={{ headerShown: false}}/>
        <Stack.Screen name="Info" component={AboutScreen} options={{ headerShown: false }}/>
        <Stack.Screen name="Results" component={ResultScreen} options={{ headerShown: false}}/>
      </Stack.Navigator>
    </NavigationContainer>
  );
}