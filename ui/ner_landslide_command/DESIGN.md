---
name: NER Landslide Command
colors:
  surface: '#10131a'
  surface-dim: '#10131a'
  surface-bright: '#363940'
  surface-container-lowest: '#0b0e14'
  surface-container-low: '#191c22'
  surface-container: '#1d2026'
  surface-container-high: '#272a31'
  surface-container-highest: '#32353c'
  on-surface: '#e1e2eb'
  on-surface-variant: '#bbcabf'
  inverse-surface: '#e1e2eb'
  inverse-on-surface: '#2e3037'
  outline: '#86948a'
  outline-variant: '#3c4a42'
  surface-tint: '#4edea3'
  primary: '#4edea3'
  on-primary: '#003824'
  primary-container: '#10b981'
  on-primary-container: '#00422b'
  inverse-primary: '#006c49'
  secondary: '#ffb95f'
  on-secondary: '#472a00'
  secondary-container: '#ee9800'
  on-secondary-container: '#5b3800'
  tertiary: '#ffb3ad'
  on-tertiary: '#68000a'
  tertiary-container: '#ff7a73'
  on-tertiary-container: '#79000e'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#6ffbbe'
  primary-fixed-dim: '#4edea3'
  on-primary-fixed: '#002113'
  on-primary-fixed-variant: '#005236'
  secondary-fixed: '#ffddb8'
  secondary-fixed-dim: '#ffb95f'
  on-secondary-fixed: '#2a1700'
  on-secondary-fixed-variant: '#653e00'
  tertiary-fixed: '#ffdad7'
  tertiary-fixed-dim: '#ffb3ad'
  on-tertiary-fixed: '#410004'
  on-tertiary-fixed-variant: '#930013'
  background: '#10131a'
  on-background: '#e1e2eb'
  surface-variant: '#32353c'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  title-lg:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
  mono-data:
    fontFamily: Courier Prime
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  gutter: 16px
  margin: 24px
---

## Brand & Style
The design system is engineered for high-stakes, real-time monitoring and rapid decision-making. It adopts a **Modern Corporate** aesthetic tailored for a **Command Center** environment, emphasizing clarity, speed of recognition, and reduced eye strain during 24/7 operations.

The visual narrative is "Precision through Data." It utilizes a dark-mode foundation to make semantic status colors (Red, Amber, Green) pop immediately against the neutral background. The style balances density with legibility, ensuring that mission-critical information is never obscured by decorative elements. The overall atmosphere is authoritative, calm, and technologically advanced.

## Colors
The palette is strictly functional, mapping color to risk level.

- **Foundational Neutrals:** The base background is a deep charcoal (`#0B0E14`). Component surfaces use a slightly lighter navy-grey (`#151921`) to create subtle visual separation without relying on heavy borders.
- **Semantic Accents:** 
  - **Safe (Primary):** Emerald Green (`#10B981`) indicates stable ground and active monitoring.
  - **Caution (Secondary):** Amber (`#F59E0B`) signals escalating risk or sensor maintenance requirements.
  - **Critical (Tertiary):** Vivid Red (`#EF4444`) is reserved exclusively for landslide alerts, evacuations, and system failures.
- **Typography:** Pure white (`#FFFFFF`) is used for headlines and critical values. A soft blue-grey (`#E2E8F0`) is used for secondary labels and metadata to maintain hierarchy.

## Typography
This design system utilizes **Inter** for its neutral, highly legible character, particularly in small sizes within data tables and dashboards. 

For telemetry data and coordinate readings, a secondary monospaced font (**Courier Prime**) should be used to ensure tabular figures align perfectly, aiding in the rapid scanning of numeric fluctuations.

**Guidelines:**
- Use `label-md` in all-caps for section headers to provide clear structure.
- Numeric values in dashboards should use `title-lg` or `headline-md` for maximum impact.
- Avoid font weights below 400 to maintain legibility on dark backgrounds.

## Layout & Spacing
The layout follows a **Fluid Grid** model with a hard 4px/8px baseline rhythm. This ensures that even high-density dashboards feel organized and intentional.

- **Desktop:** 12-column grid. Sidebars are fixed at 280px. Main content areas use a 24px margin.
- **Tablet:** 8-column grid. Sidebars collapse into icons or a hamburger menu.
- **Mobile:** 4-column grid. Charts and data cards stack vertically.

Spacing between related data points should use `sm` (8px), while distinct functional groups or cards should use `lg` (24px) to provide visual "breathing room" in complex interfaces.

## Elevation & Depth
In this dark-mode environment, depth is established through **Tonal Layers** and subtle luminosity rather than traditional heavy shadows.

- **Level 0 (Base):** `#0B0E14` – Used for the global background.
- **Level 1 (Card/Surface):** `#151921` – Used for primary content containers. These should have a very soft, diffuse shadow (`0px 4px 20px rgba(0,0,0,0.5)`).
- **Level 2 (Active/Hover):** A thin 1px border using `#E2E8F0` at 10% opacity is added to cards to define edges against the dark background.
- **Level 3 (Modals/Popovers):** Higher elevation with a 20% opacity black shadow and a subtle inner glow on the top edge to simulate a light source from above.

## Shapes
The shape language is "Approachable Professionalism." While the data is clinical, the container edges are softened to reduce the perceived harshness of the dark-mode UI.

- **Cards & Major Containers:** 16px (`rounded-lg`) corner radius.
- **Buttons & Inputs:** 8px corner radius for a sturdy, reliable feel.
- **Status Indicators (Pills):** Fully rounded (pill-shaped) to distinguish them from interactive buttons.

## Components
- **Buttons:** Primary buttons use a solid `#10B981` with white text. Secondary buttons use an outlined style (1px border) with `#E2E8F0`. High-alert buttons (Stop/Clear) use the tertiary Red.
- **Cards:** Dashboard cards must include a 16px padding (`md`). Headers within cards should be separated by a subtle 1px divider.
- **Input Fields:** Background should be slightly darker than the card surface (`#0B0E14`). Borders are invisible until focus, at which point they glow with a 1px Primary color stroke.
- **Status Chips:** Small, high-contrast badges. Use a background with 15% opacity of the semantic color and a 100% opacity text color (e.g., Light Red text on Dark Red background).
- **Data Visualizations:** Use simple line icons (2px stroke). Charts should use the primary/secondary/tertiary colors for trend lines. Grids within charts should be kept at 5% opacity to avoid visual noise.
- **Alert Banner:** Fixed to the top of the viewport. High-risk alerts should pulse subtly between `#EF4444` and a darker shade to command attention without being distracting.