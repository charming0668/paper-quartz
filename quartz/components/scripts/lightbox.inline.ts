/**
 * lightbox.inline.ts
 * 纯原生、高性能、无依赖的学术文献图片放大查看器 (Image Lightbox)
 * 适配 Quartz SPA 导航，支持 ESC 退出、遮罩层点击退出、图注同步显示与滚动/全屏适配
 */

let overlayEl: HTMLElement | null = null
let imgEl: HTMLImageElement | null = null
let captionEl: HTMLElement | null = null

function createLightboxDOM() {
  if (overlayEl) return

  overlayEl = document.createElement("div")
  overlayEl.className = "quartz-lightbox-overlay"
  overlayEl.setAttribute("aria-hidden", "true")
  overlayEl.setAttribute("role", "dialog")
  overlayEl.setAttribute("aria-label", "图片预览放大")

  overlayEl.innerHTML = `
    <div class="quartz-lightbox-backdrop"></div>
    <div class="quartz-lightbox-container">
      <button class="quartz-lightbox-close" aria-label="关闭预览" title="关闭 (ESC)">
        <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
      <div class="quartz-lightbox-figure">
        <img class="quartz-lightbox-img" src="" alt="" />
        <div class="quartz-lightbox-caption"></div>
      </div>
      <div class="quartz-lightbox-hint">ESC 退出 · 点击空白处关闭</div>
    </div>
  `

  document.body.appendChild(overlayEl)

  imgEl = overlayEl.querySelector(".quartz-lightbox-img")
  captionEl = overlayEl.querySelector(".quartz-lightbox-caption")

  const backdrop = overlayEl.querySelector(".quartz-lightbox-backdrop")
  const closeBtn = overlayEl.querySelector(".quartz-lightbox-close")

  const closeHandler = () => closeLightbox()
  backdrop?.addEventListener("click", closeHandler)
  closeBtn?.addEventListener("click", closeHandler)

  // 点击 container 空白区域也退出，但点击图片本身不退出
  const container = overlayEl.querySelector(".quartz-lightbox-container")
  container?.addEventListener("click", (e) => {
    if (e.target === container || (e.target as HTMLElement).classList.contains("quartz-lightbox-figure")) {
      closeLightbox()
    }
  })

  // 键盘 ESC 监听
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlayEl?.classList.contains("active")) {
      closeLightbox()
    }
  })
}

function openLightbox(src: string, alt: string, captionText: string) {
  if (!overlayEl || !imgEl) createLightboxDOM()
  if (!overlayEl || !imgEl) return

  imgEl.src = src
  imgEl.alt = alt

  if (captionEl) {
    if (captionText) {
      captionEl.textContent = captionText
      captionEl.style.display = "block"
    } else {
      captionEl.style.display = "none"
    }
  }

  overlayEl.style.display = "flex"
  // 强制触发 reflow 以启用 transition 动画
  void overlayEl.offsetWidth
  overlayEl.classList.add("active")
  overlayEl.setAttribute("aria-hidden", "false")
  document.body.classList.add("quartz-lightbox-open")
}

function closeLightbox() {
  if (!overlayEl) return
  overlayEl.classList.remove("active")
  overlayEl.setAttribute("aria-hidden", "true")
  document.body.classList.remove("quartz-lightbox-open")

  setTimeout(() => {
    if (!overlayEl?.classList.contains("active")) {
      overlayEl!.style.display = "none"
      if (imgEl) imgEl.src = ""
    }
  }, 250)
}

function setupLightbox() {
  createLightboxDOM()

  // 获取正文 article 中的所有插图
  const images = document.querySelectorAll<HTMLImageElement>("article img")
  images.forEach((img) => {
    // 避免重复绑定
    if (img.dataset.lightboxBound === "true") return
    img.dataset.lightboxBound = "true"
    img.classList.add("quartz-zoomable-img")
    img.setAttribute("title", "点击放大查看高清大图")

    img.addEventListener("click", (e) => {
      e.stopPropagation()
      const src = img.currentSrc || img.src

      // 提取图注：查找当前 p 标签后的 em，或者紧随其后的 em/blockquote
      let caption = img.alt || ""
      const parentP = img.closest("p")
      if (parentP) {
        const nextEl = parentP.nextElementSibling
        const emInP = parentP.querySelector("em")
        if (emInP) {
          caption = emInP.textContent?.trim() || caption
        } else if (nextEl && nextEl.tagName.toLowerCase() === "p" && nextEl.querySelector("em")) {
          caption = nextEl.querySelector("em")?.textContent?.trim() || caption
        }
      }

      openLightbox(src, img.alt, caption)
    })
  })
}

// 绑定 SPA 路由与首次加载事件
document.addEventListener("nav", setupLightbox)
document.addEventListener("DOMContentLoaded", setupLightbox)
setupLightbox()

