const currentUser = document.querySelector("#currentUser");
const logoutLink = document.querySelector("#logoutLink");
const analysisTab = document.querySelector("#analysisTab");
const adminTab = document.querySelector("#adminTab");
const analysisView = document.querySelector("#analysisView");
const adminView = document.querySelector("#adminView");

const form = document.querySelector("#uploadForm");
const imageInput = document.querySelector("#imageInput");
const fileName = document.querySelector("#fileName");
const submitButton = document.querySelector("#submitButton");
const detectionCount = document.querySelector("#detectionCount");
const areaValue = document.querySelector("#areaValue");
const ratioValue = document.querySelector("#ratioValue");
const severityValue = document.querySelector("#severityValue");
const annotatedImage = document.querySelector("#annotatedImage");
const maskImage = document.querySelector("#maskImage");
const detectionRows = document.querySelector("#detectionRows");
const jsonOutput = document.querySelector("#jsonOutput");

const createUserForm = document.querySelector("#createUserForm");
const createUserButton = document.querySelector("#createUserButton");
const newUsername = document.querySelector("#newUsername");
const newPassword = document.querySelector("#newPassword");
const newRole = document.querySelector("#newRole");
const adminMessage = document.querySelector("#adminMessage");
const userRows = document.querySelector("#userRows");

let sessionUser = null;

document.addEventListener("DOMContentLoaded", restoreSession);

logoutLink.addEventListener("click", () => {
  sessionUser = null;
});

analysisTab.addEventListener("click", () => setActiveTab("analysis"));
adminTab.addEventListener("click", () => setActiveTab("admin"));

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  fileName.textContent = file ? file.name : "Belum ada file";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const mmPerPixel = document.querySelector("#mmPerPixel").value;
  if (mmPerPixel) {
    formData.append("mm_per_pixel", mmPerPixel);
  }

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Gagal menganalisis gambar.");
    }

    renderResult(payload);
  } catch (error) {
    jsonOutput.textContent = JSON.stringify({ error: error.message }, null, 2);
  } finally {
    setLoading(false);
  }
});

createUserForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  adminMessage.textContent = "";
  createUserButton.disabled = true;

  try {
    const response = await fetch("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: newUsername.value,
        password: newPassword.value,
        role: newRole.value,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Gagal menambah user.");
    createUserForm.reset();
    adminMessage.textContent = `User ${payload.username} berhasil ditambahkan.`;
    await loadUsers();
  } catch (error) {
    adminMessage.textContent = error.message;
  } finally {
    createUserButton.disabled = false;
  }
});

async function restoreSession() {
  try {
    const response = await fetch("/api/me");
    if (!response.ok) {
      window.location.href = "/";
      return;
    }
    const payload = await response.json();
    showApp(payload);
  } catch {
    window.location.href = "/";
  }
}

function showApp(user) {
  sessionUser = user;
  currentUser.textContent = `${user.username} (${user.role})`;
  adminTab.classList.toggle("hidden", user.role !== "admin");
  setActiveTab("analysis");
  if (user.role === "admin") {
    loadUsers();
  }
}

function setActiveTab(tab) {
  const isAdminTab = tab === "admin";
  analysisView.classList.toggle("hidden", isAdminTab);
  adminView.classList.toggle("hidden", !isAdminTab);
  analysisTab.classList.toggle("active", !isAdminTab);
  adminTab.classList.toggle("active", isAdminTab);

  if (isAdminTab && sessionUser?.role === "admin") {
    loadUsers();
  }
}

async function loadUsers() {
  adminMessage.textContent = "Memuat daftar user...";
  try {
    const response = await fetch("/api/users");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Gagal mengambil daftar user.");
    }

    userRows.innerHTML = "";
    for (const user of payload) {
      const tr = document.createElement("tr");
      const canDelete = user.username !== sessionUser.username;
      tr.innerHTML = `
        <td>
          <div class="user-cell">
            <strong>${escapeHtml(user.username)}</strong>
            <span>${user.username === sessionUser.username ? "Akun aktif" : "Terdaftar"}</span>
          </div>
        </td>
        <td><span class="role-badge">${user.role}</span></td>
        <td>
          <button class="danger-button" data-username="${escapeHtml(user.username)}" ${canDelete ? "" : "disabled"}>
            Hapus
          </button>
        </td>
      `;
      userRows.appendChild(tr);
    }

    userRows.querySelectorAll("button[data-username]").forEach((button) => {
      button.addEventListener("click", () => deleteUser(button.dataset.username));
    });
    adminMessage.textContent = `${payload.length} user terdaftar.`;
  } catch (error) {
    adminMessage.textContent = error.message;
  }
}

async function deleteUser(username) {
  adminMessage.textContent = "";
  const response = await fetch(`/api/users/${encodeURIComponent(username)}`, {
    method: "DELETE",
  });
  const payload = await response.json();
  if (!response.ok) {
    adminMessage.textContent = payload.detail || "Gagal menghapus user.";
    return;
  }
  adminMessage.textContent = `User ${username} berhasil dihapus.`;
  await loadUsers();
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Menganalisis..." : "Analisis Gambar";
}

function renderResult(payload) {
  detectionCount.textContent = payload.summary.detections;
  areaValue.textContent = formatArea(payload.summary);
  ratioValue.textContent = `${(payload.summary.corrosion_ratio * 100).toFixed(2)}%`;
  severityValue.textContent = payload.summary.severity;

  annotatedImage.src = payload.artifacts.annotated_image_url || "";
  maskImage.src = payload.artifacts.mask_image_url || "";

  detectionRows.innerHTML = "";
  for (const detection of payload.detections) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${detection.id}</td>
      <td>${(detection.confidence * 100).toFixed(1)}%</td>
      <td>${detection.bbox_xyxy.join(", ")}</td>
      <td>${formatDetectionArea(detection)}</td>
      <td>${(detection.corrosion_ratio_in_bbox * 100).toFixed(2)}%</td>
      <td>${detection.severity}</td>
    `;
    detectionRows.appendChild(tr);
  }

  if (payload.detections.length === 0) {
    const tr = document.createElement("tr");
    tr.innerHTML = '<td colspan="6">Tidak ada korosi terdeteksi.</td>';
    detectionRows.appendChild(tr);
  }

  jsonOutput.textContent = JSON.stringify(payload, null, 2);
}

function formatArea(summary) {
  if (summary.total_corrosion_area_cm2 !== null) {
    return `${summary.total_corrosion_area_cm2.toFixed(2)} cm2`;
  }
  return `${summary.total_corrosion_area_px} px`;
}

function formatDetectionArea(detection) {
  if (detection.corrosion_area_cm2 !== null) {
    return `${detection.corrosion_area_cm2.toFixed(2)} cm2`;
  }
  return `${detection.corrosion_area_px} px`;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}
