import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";

const mount = document.querySelector("#motorScene");
if (mount) {
  try {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
    camera.position.set(5.8, 4.2, 7.6);
    camera.lookAt(0, 0.15, 0);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xdce7ff, 0x15214d, 2.1));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
    keyLight.position.set(3, 7, 5);
    scene.add(keyLight);
    const rimLight = new THREE.PointLight(0x5c8dff, 38, 14);
    rimLight.position.set(-4, 2, -3);
    scene.add(rimLight);

    const materials = {
      body: new THREE.MeshStandardMaterial({ color: 0x426be0, roughness: 0.34, metalness: 0.42 }),
      dark: new THREE.MeshStandardMaterial({ color: 0x172b69, roughness: 0.4, metalness: 0.48 }),
      steel: new THREE.MeshStandardMaterial({ color: 0xb6c5e8, roughness: 0.25, metalness: 0.82 }),
      bright: new THREE.MeshStandardMaterial({ color: 0x80a9ff, roughness: 0.3, metalness: 0.55 }),
      rubber: new THREE.MeshStandardMaterial({ color: 0x263765, roughness: 0.58, metalness: 0.18 }),
      glow: new THREE.MeshStandardMaterial({ color: 0x50d4cb, emissive: 0x159b9e, emissiveIntensity: 1.25, roughness: 0.24, metalness: 0.48 }),
    };

    const machine = new THREE.Group();
    machine.rotation.y = -0.42;
    machine.rotation.x = -0.07;
    scene.add(machine);

    const addBox = (parent, size, position, material, bevel = false) => {
      const geo = new THREE.BoxGeometry(...size);
      const mesh = new THREE.Mesh(geo, material);
      mesh.position.set(...position);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      parent.add(mesh);
      return mesh;
    };
    const addCylinder = (parent, radiusTop, radiusBottom, length, position, material, axis = "x", segments = 48) => {
      const geo = new THREE.CylinderGeometry(radiusTop, radiusBottom, length, segments);
      const mesh = new THREE.Mesh(geo, material);
      if (axis === "x") mesh.rotation.z = Math.PI / 2;
      mesh.position.set(...position);
      mesh.castShadow = true;
      parent.add(mesh);
      return mesh;
    };

    // Wide foot plate and paired mounting skids.
    addBox(machine, [3.9, 0.18, 2.35], [0, -1.05, 0], materials.dark);
    addBox(machine, [3.45, 0.12, 1.9], [0, -0.9, 0], materials.steel);
    for (const z of [-0.72, 0.72]) {
      addBox(machine, [3.3, 0.18, 0.22], [0, -0.78, z], materials.rubber);
      for (const x of [-1.36, 1.36]) {
        addCylinder(machine, 0.075, 0.075, 0.05, [x, -0.66, z], materials.bright, "y", 16);
      }
    }

    // Motor casing with a raised terminal box and restrained cooling ribs.
    addBox(machine, [2.18, 1.48, 1.42], [-0.45, 0, 0], materials.body);
    addBox(machine, [1.58, 0.13, 1.1], [-0.48, 0.8, 0], materials.dark);
    addBox(machine, [0.74, 0.43, 0.63], [-0.6, 1.06, 0], materials.dark);
    addBox(machine, [0.51, 0.05, 0.44], [-0.6, 1.3, 0], materials.bright);
    for (let i = 0; i < 7; i++) {
      addBox(machine, [0.08, 1.05, 1.52], [-1.36 + i * 0.27, -0.02, 0], i % 2 ? materials.dark : materials.bright);
    }
    addBox(machine, [0.12, 0.96, 1.56], [0.7, -0.02, 0], materials.dark);

    // End bell, bearing collar, shaft and polished exposed rotor.
    addCylinder(machine, 0.78, 0.78, 0.23, [0.78, 0, 0], materials.dark);
    addCylinder(machine, 0.58, 0.58, 0.14, [0.94, 0, 0], materials.bright);
    addCylinder(machine, 0.35, 0.35, 0.13, [1.07, 0, 0], materials.glow);
    addCylinder(machine, 0.16, 0.16, 0.76, [1.46, 0, 0], materials.steel);
    addCylinder(machine, 0.21, 0.21, 0.16, [1.88, 0, 0], materials.bright);
    const rotor = new THREE.Group();
    rotor.position.set(1.57, 0, 0);
    machine.add(rotor);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.38, 0.035, 12, 64), materials.glow);
    ring.rotation.y = Math.PI / 2;
    rotor.add(ring);
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      const spoke = addBox(rotor, [0.42, 0.035, 0.035], [Math.cos(angle) * 0.19, Math.sin(angle) * 0.19, 0], materials.steel);
      spoke.rotation.z = angle;
    }

    // Sparse mounting bolts and side inspection panel.
    addBox(machine, [0.7, 0.58, 0.045], [-0.48, -0.03, 0.735], materials.dark);
    addBox(machine, [0.55, 0.43, 0.04], [-0.48, -0.03, 0.77], materials.bright);
    for (const x of [-0.7, -0.25]) for (const y of [-0.2, 0.14]) {
      addCylinder(machine, 0.035, 0.035, 0.035, [x, y, 0.81], materials.steel, "z", 12);
    }
    const axisLine = new THREE.Mesh(
      new THREE.CylinderGeometry(0.012, 0.012, 4.9, 8),
      new THREE.MeshBasicMaterial({ color: 0x91b1ff, transparent: true, opacity: 0.48 })
    );
    axisLine.rotation.z = Math.PI / 2;
    axisLine.position.set(0.3, 0.06, 0);
    machine.add(axisLine);

    const resize = new ResizeObserver(() => {
      const { width, height } = mount.getBoundingClientRect();
      if (!width || !height) return;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    });
    resize.observe(mount);

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let frame;
    const render = (time = 0) => {
      frame = requestAnimationFrame(render);
      if (!reduceMotion) {
        machine.rotation.y = -0.42 + Math.sin(time * 0.00032) * 0.13;
        machine.position.y = Math.sin(time * 0.0008) * 0.035;
        rotor.rotation.x = time * 0.0011;
      }
      renderer.render(scene, camera);
    };
    render();
    window.addEventListener("pagehide", () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      renderer.dispose();
    }, { once: true });
  } catch (error) {
    console.warn("[VibraGuard] 3D motor view unavailable:", error);
    mount.classList.add("scene-unavailable");
  }
}
