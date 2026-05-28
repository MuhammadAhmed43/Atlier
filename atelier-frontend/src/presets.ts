export type PresetMap = Record<string, Record<string, string>>;

export const PRESETS: PresetMap = {
  Outerwear: {
    'Navy Wool Blazer': 'A navy wool blazer with gold buttons, structural shoulders, heavy fabric',
    'Black Leather Jacket':
      'A classic black leather jacket with silver zippers and fitted silhouette',
    'Brown Bomber Jacket': 'A brown leather bomber jacket with shearling collar and ribbed cuffs',
    'Denim Jacket': 'A vintage medium-wash denim jacket with brass buttons',
    'Wool Overcoat': 'An elegant charcoal wool overcoat with notched lapels',
    'Beige Trench Coat': 'A structured beige trench coat with belted waist and storm flap',
  },
  Tops: {
    'White Linen Shirt': 'A crisp white linen shirt with relaxed fit and clean lines',
    'Silk Blouse': 'An elegant cream silk blouse with subtle sheen',
    'Cashmere Sweater': 'A cozy beige cashmere sweater with ribbed texture and soft finish',
    'Striped T-Shirt': 'A classic navy and white striped cotton t-shirt with crew neck',
    'Wool Cardigan': 'A chunky cream wool cardigan with wooden buttons',
  },
  Dresses: {
    'Red Silk Dress': 'A bright red silk evening dress with elegant draping and flowing fabric',
    'Velvet Cocktail': 'A deep burgundy velvet cocktail dress with subtle sheen',
    'Little Black Dress': 'A sophisticated black fitted dress with clean lines and minimal design',
    'Floral Sundress': 'A light floral print sundress with flowing skirt',
  },
  Bottoms: {
    'Dark Indigo Jeans': 'Classic dark indigo straight-leg denim jeans with clean stitching',
    'Tailored Trousers': 'Elegant black tailored wool trousers with sharp creases',
    'Leather Pants': 'Sleek black leather pants with slim fit',
  },
};

export const CATEGORIES = Object.keys(PRESETS);

export function randomPreset(): { category: string; name: string; prompt: string } {
  const category = CATEGORIES[Math.floor(Math.random() * CATEGORIES.length)];
  const presets = PRESETS[category];
  const names = Object.keys(presets);
  const name = names[Math.floor(Math.random() * names.length)];
  return { category, name, prompt: presets[name] };
}
