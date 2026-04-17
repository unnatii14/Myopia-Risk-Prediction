import { createBrowserRouter } from "react-router";
import Landing from "./pages/Landing";
import Screen from "./pages/Screen";
import Results from "./pages/Results";
import Progression from "./pages/Progression";
import AxialElongation from "./pages/AxialElongation";
import OnsetPredictor from "./pages/OnsetPredictor";
import ImagePredictor from "./pages/ImagePredictor";
import FAQ from "./pages/FAQ";
import About from "./pages/About";
import NotFound from "./pages/NotFound";
import RootLayout from "./layouts/RootLayout";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import PrivateRoute from "./components/PrivateRoute";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: RootLayout,
    children: [
      {
        index: true,
        Component: Landing,
      },
      // Protected — must be logged in
      {
        Component: PrivateRoute,
        children: [
          {
            path: "screen",
            Component: Screen,
          },
          {
            path: "results",
            Component: Results,
          },
          {
            path: "progression",
            Component: Progression,
          },
          {
            path: "axial",
            Component: AxialElongation,
          },
          {
            path: "onset",
            Component: OnsetPredictor,
          },
          {
            path: "image-predictor",
            Component: ImagePredictor,
          },
        ],
      },
      {
        path: "faq",
        Component: FAQ,
      },
      {
        path: "about",
        Component: About,
      },
      {
        path: "login",
        Component: Login,
      },
      {
        path: "signup",
        Component: Signup,
      },
      {
        path: "*",
        Component: NotFound,
      },
    ],
  },
]);
