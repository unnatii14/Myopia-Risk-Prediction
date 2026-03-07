import { createBrowserRouter } from "react-router";
import Landing from "./pages/Landing";
import Screen from "./pages/Screen";
import Results from "./pages/Results";
import FAQ from "./pages/FAQ";
import About from "./pages/About";
import NotFound from "./pages/NotFound";
import RootLayout from "./layouts/RootLayout";
import Login from "./pages/Login";
import Signup from "./pages/Signup";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: RootLayout,
    children: [
      {
        index: true,
        Component: Landing,
      },
      {
        path: "screen",
        Component: Screen,
      },
      {
        path: "results",
        Component: Results,
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