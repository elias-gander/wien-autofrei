let isCarsPresented = false;
const viewer = pannellum.viewer("panorama", {
  default: {
    firstScene: "cars",
    sceneFadeDuration: 1000,
  },
  scenes: {
    cars: {
      type: "equirectangular",
      panorama: "../panorama.jpg",
    },
    carfree: {
      type: "equirectangular",
      panorama: "../panorama-carfree.jpg",
    },
  },
});
viewer.loadScene("carfree");

document.getElementById("toggle-button").addEventListener("click", () => {
  const yaw = viewer.getYaw();
  const pitch = viewer.getPitch();
  const hfov = viewer.getHfov();
  if (isCarsPresented) {
    viewer.loadScene("carfree");
  } else {
    viewer.loadScene("cars");
  }
  isCarsPresented = !isCarsPresented;
  setTimeout(() => {
    viewer.setYaw(yaw);
    viewer.setPitch(pitch);
    viewer.setHfov(hfov);
  }, 50);
});
