from __future__ import annotations

# Gen 3 (Pokemon FireRed / pokefirered) internal type constants
TYPE_NORMAL = 0
TYPE_FIGHTING = 1
TYPE_FLYING = 2
TYPE_POISON = 3
TYPE_GROUND = 4
TYPE_ROCK = 5
TYPE_BUG = 6
TYPE_GHOST = 7
TYPE_STEEL = 8
TYPE_MYSTERY = 9
TYPE_FIRE = 10
TYPE_WATER = 11
TYPE_GRASS = 12
TYPE_ELECTRIC = 13
TYPE_PSYCHIC = 14
TYPE_ICE = 15
TYPE_DRAGON = 16
TYPE_DARK = 17

TYPE_CHART: dict[int, dict[int, float]] = {
    TYPE_NORMAL:   {TYPE_ROCK: 0.5, TYPE_STEEL: 0.5, TYPE_GHOST: 0.0},
    TYPE_FIGHTING: {
        TYPE_NORMAL: 2.0, TYPE_ICE: 2.0, TYPE_ROCK: 2.0, TYPE_DARK: 2.0, TYPE_STEEL: 2.0,
        TYPE_POISON: 0.5, TYPE_FLYING: 0.5, TYPE_PSYCHIC: 0.5, TYPE_BUG: 0.5, TYPE_GHOST: 0.0,
    },
    TYPE_FLYING:   {
        TYPE_GRASS: 2.0, TYPE_FIGHTING: 2.0, TYPE_BUG: 2.0,
        TYPE_ELECTRIC: 0.5, TYPE_ROCK: 0.5, TYPE_STEEL: 0.5,
    },
    TYPE_POISON:   {
        TYPE_GRASS: 2.0,
        TYPE_POISON: 0.5, TYPE_GROUND: 0.5, TYPE_ROCK: 0.5, TYPE_GHOST: 0.5, TYPE_STEEL: 0.0,
    },
    TYPE_GROUND:   {
        TYPE_FIRE: 2.0, TYPE_ELECTRIC: 2.0, TYPE_POISON: 2.0, TYPE_ROCK: 2.0, TYPE_STEEL: 2.0,
        TYPE_GRASS: 0.5, TYPE_BUG: 0.5, TYPE_FLYING: 0.0,
    },
    TYPE_ROCK:     {
        TYPE_FIRE: 2.0, TYPE_ICE: 2.0, TYPE_FLYING: 2.0, TYPE_BUG: 2.0,
        TYPE_FIGHTING: 0.5, TYPE_GROUND: 0.5, TYPE_STEEL: 0.5,
    },
    TYPE_BUG:      {
        TYPE_GRASS: 2.0, TYPE_PSYCHIC: 2.0, TYPE_DARK: 2.0,
        TYPE_FIRE: 0.5, TYPE_FIGHTING: 0.5, TYPE_POISON: 0.5,
        TYPE_FLYING: 0.5, TYPE_GHOST: 0.5, TYPE_STEEL: 0.5,
    },
    TYPE_GHOST:    {TYPE_PSYCHIC: 2.0, TYPE_GHOST: 2.0, TYPE_DARK: 0.5, TYPE_STEEL: 0.5, TYPE_NORMAL: 0.0},
    TYPE_STEEL:    {TYPE_ICE: 2.0, TYPE_ROCK: 2.0, TYPE_FIRE: 0.5, TYPE_WATER: 0.5, TYPE_ELECTRIC: 0.5, TYPE_STEEL: 0.5},
    TYPE_FIRE:     {
        TYPE_GRASS: 2.0, TYPE_ICE: 2.0, TYPE_BUG: 2.0, TYPE_STEEL: 2.0,
        TYPE_FIRE: 0.5, TYPE_WATER: 0.5, TYPE_ROCK: 0.5, TYPE_DRAGON: 0.5,
    },
    TYPE_WATER:    {
        TYPE_FIRE: 2.0, TYPE_GROUND: 2.0, TYPE_ROCK: 2.0,
        TYPE_WATER: 0.5, TYPE_GRASS: 0.5, TYPE_DRAGON: 0.5,
    },
    TYPE_GRASS:    {
        TYPE_WATER: 2.0, TYPE_GROUND: 2.0, TYPE_ROCK: 2.0,
        TYPE_FIRE: 0.5, TYPE_GRASS: 0.5, TYPE_POISON: 0.5, TYPE_FLYING: 0.5,
        TYPE_BUG: 0.5, TYPE_DRAGON: 0.5, TYPE_STEEL: 0.5,
    },
    TYPE_ELECTRIC: {
        TYPE_WATER: 2.0, TYPE_FLYING: 2.0,
        TYPE_GRASS: 0.5, TYPE_ELECTRIC: 0.5, TYPE_DRAGON: 0.5, TYPE_GROUND: 0.0,
    },
    TYPE_PSYCHIC:  {TYPE_FIGHTING: 2.0, TYPE_POISON: 2.0, TYPE_PSYCHIC: 0.5, TYPE_STEEL: 0.5, TYPE_DARK: 0.0},
    TYPE_ICE:      {
        TYPE_GRASS: 2.0, TYPE_GROUND: 2.0, TYPE_FLYING: 2.0, TYPE_DRAGON: 2.0,
        TYPE_FIRE: 0.5, TYPE_WATER: 0.5, TYPE_ICE: 0.5, TYPE_STEEL: 0.5,
    },
    TYPE_DRAGON:   {TYPE_DRAGON: 2.0, TYPE_STEEL: 0.5},
    TYPE_DARK:     {TYPE_PSYCHIC: 2.0, TYPE_GHOST: 2.0, TYPE_FIGHTING: 0.5, TYPE_DARK: 0.5, TYPE_STEEL: 0.5},
}

# move_id -> (type_id, base_power)
MOVES_DB: dict[int, tuple[int, int]] = {
    1: (TYPE_NORMAL, 40),    # Pound
    2: (TYPE_FIGHTING, 50),  # Karate Chop
    3: (TYPE_NORMAL, 30),    # Double Slap (multi-hit avg)
    4: (TYPE_NORMAL, 36),    # Comet Punch
    5: (TYPE_NORMAL, 80),    # Mega Punch
    6: (TYPE_NORMAL, 40),    # Pay Day
    7: (TYPE_FIRE, 75),      # Fire Punch
    8: (TYPE_ICE, 75),       # Ice Punch
    9: (TYPE_ELECTRIC, 75),  # Thunder Punch
    10: (TYPE_NORMAL, 40),   # Scratch
    11: (TYPE_NORMAL, 55),   # ViceGrip
    13: (TYPE_NORMAL, 80),   # Razor Wind
    14: (TYPE_NORMAL, 0),    # Swords Dance
    15: (TYPE_NORMAL, 50),   # Cut
    16: (TYPE_FLYING, 40),   # Gust
    17: (TYPE_FLYING, 60),   # Wing Attack
    18: (TYPE_NORMAL, 0),    # Whirlwind
    19: (TYPE_FLYING, 70),   # Fly
    20: (TYPE_NORMAL, 15),   # Bind
    21: (TYPE_NORMAL, 80),   # Slam
    22: (TYPE_GRASS, 35),    # Vine Whip
    23: (TYPE_NORMAL, 65),   # Stomp
    24: (TYPE_FIGHTING, 60), # Double Kick (2 hits)
    25: (TYPE_NORMAL, 120),  # Mega Kick
    26: (TYPE_FIGHTING, 70), # Jump Kick
    27: (TYPE_FIGHTING, 60), # Rolling Kick
    28: (TYPE_GROUND, 0),    # Sand Attack
    29: (TYPE_NORMAL, 70),   # Headbutt
    30: (TYPE_NORMAL, 65),   # Horn Attack
    31: (TYPE_NORMAL, 30),   # Fury Attack
    33: (TYPE_NORMAL, 35),   # Tackle (Gen 3 = 35)
    34: (TYPE_NORMAL, 85),   # Body Slam
    35: (TYPE_NORMAL, 15),   # Wrap
    36: (TYPE_NORMAL, 90),   # Take Down
    37: (TYPE_NORMAL, 90),   # Thrash
    38: (TYPE_NORMAL, 120),  # Double-Edge
    39: (TYPE_NORMAL, 0),    # Tail Whip
    40: (TYPE_POISON, 15),   # Poison Sting
    41: (TYPE_BUG, 50),      # Twineedle (2 hits)
    42: (TYPE_BUG, 28),      # Pin Missile
    43: (TYPE_NORMAL, 0),    # Leer
    44: (TYPE_DARK, 60),     # Bite
    45: (TYPE_NORMAL, 0),    # Growl
    46: (TYPE_NORMAL, 0),    # Roar
    47: (TYPE_NORMAL, 0),    # Sing
    48: (TYPE_NORMAL, 0),    # Supersonic
    49: (TYPE_NORMAL, 20),   # SonicBoom
    50: (TYPE_NORMAL, 0),    # Disable
    51: (TYPE_POISON, 40),   # Acid
    52: (TYPE_FIRE, 40),     # Ember
    53: (TYPE_FIRE, 95),     # Flamethrower
    54: (TYPE_ICE, 0),       # Mist
    55: (TYPE_WATER, 40),    # Water Gun
    56: (TYPE_WATER, 120),   # Hydro Pump
    57: (TYPE_WATER, 95),    # Surf
    58: (TYPE_ICE, 95),      # Ice Beam
    59: (TYPE_ICE, 120),   # Blizzard
    60: (TYPE_PSYCHIC, 65),  # Psybeam
    61: (TYPE_WATER, 65),    # Bubble Beam
    62: (TYPE_ICE, 65),      # Aurora Beam
    63: (TYPE_NORMAL, 150),  # Hyper Beam
    64: (TYPE_FLYING, 35),   # Peck
    65: (TYPE_FLYING, 80),   # Drill Peck
    66: (TYPE_FIGHTING, 80), # Submission
    67: (TYPE_FIGHTING, 50), # Low Kick
    68: (TYPE_FIGHTING, 0),  # Counter
    69: (TYPE_FIGHTING, 50), # Seismic Toss
    70: (TYPE_NORMAL, 80),   # Strength
    71: (TYPE_GRASS, 20),    # Absorb
    72: (TYPE_GRASS, 40),    # Mega Drain
    73: (TYPE_GRASS, 0),     # Leech Seed
    74: (TYPE_NORMAL, 0),    # Growth
    75: (TYPE_GRASS, 55),    # Razor Leaf
    76: (TYPE_GRASS, 120),   # Solar Beam
    77: (TYPE_POISON, 0),    # Poison Powder
    78: (TYPE_GRASS, 0),     # Stun Spore
    79: (TYPE_GRASS, 0),     # Sleep Powder
    80: (TYPE_GRASS, 70),    # Petal Dance
    81: (TYPE_BUG, 0),       # String Shot
    82: (TYPE_DRAGON, 40),   # Dragon Rage
    83: (TYPE_FIRE, 15),     # Fire Spin
    84: (TYPE_ELECTRIC, 40), # Thunder Shock
    85: (TYPE_ELECTRIC, 95), # Thunderbolt
    86: (TYPE_ELECTRIC, 0),  # Thunder Wave
    87: (TYPE_ELECTRIC, 120),# Thunder
    88: (TYPE_ROCK, 50),     # Rock Throw
    89: (TYPE_GROUND, 100),  # Earthquake
    91: (TYPE_GROUND, 60),   # Dig
    92: (TYPE_POISON, 0),    # Toxic
    93: (TYPE_PSYCHIC, 50),  # Confusion
    94: (TYPE_PSYCHIC, 90),  # Psychic
    95: (TYPE_PSYCHIC, 0),   # Hypnosis
    96: (TYPE_PSYCHIC, 0),   # Meditate
    97: (TYPE_PSYCHIC, 0),   # Agility
    98: (TYPE_NORMAL, 40),   # Quick Attack
    99: (TYPE_NORMAL, 20),   # Rage
    100: (TYPE_PSYCHIC, 0),  # Teleport
    101: (TYPE_GHOST, 40),   # Night Shade
    103: (TYPE_NORMAL, 0),   # Screech
    104: (TYPE_NORMAL, 0),   # Double Team
    105: (TYPE_NORMAL, 0),   # Recover
    106: (TYPE_NORMAL, 0),   # Harden
    107: (TYPE_NORMAL, 0),   # Minimize
    108: (TYPE_NORMAL, 0),   # Smokescreen
    109: (TYPE_GHOST, 0),    # Confuse Ray
    110: (TYPE_WATER, 0),    # Withdraw
    111: (TYPE_NORMAL, 0),   # Defense Curl
    112: (TYPE_PSYCHIC, 0),  # Barrier
    113: (TYPE_PSYCHIC, 0),  # Light Screen
    114: (TYPE_ICE, 0),      # Haze
    115: (TYPE_PSYCHIC, 0),  # Reflect
    116: (TYPE_NORMAL, 0),   # Focus Energy
    120: (TYPE_NORMAL, 200), # Self-Destruct
    122: (TYPE_GHOST, 20),   # Lick
    123: (TYPE_POISON, 20),  # Smog
    124: (TYPE_POISON, 65),  # Sludge
    125: (TYPE_GROUND, 65),  # Bone Club
    126: (TYPE_FIRE, 120),   # Fire Blast
    127: (TYPE_WATER, 80),   # Waterfall
    128: (TYPE_WATER, 35),   # Clamp
    129: (TYPE_NORMAL, 60),  # Swift
    130: (TYPE_NORMAL, 100), # Skull Bash
    136: (TYPE_FIGHTING, 85),# Hi Jump Kick
    137: (TYPE_NORMAL, 0),   # Glare
    138: (TYPE_PSYCHIC, 100),# Dream Eater
    140: (TYPE_NORMAL, 30),  # Barrage
    141: (TYPE_BUG, 20),     # Leech Life
    145: (TYPE_WATER, 20),   # Bubble
    146: (TYPE_NORMAL, 70),  # Dizzy Punch
    147: (TYPE_GRASS, 0),    # Spore
    148: (TYPE_NORMAL, 0),   # Flash
    152: (TYPE_WATER, 90),   # Crabhammer
    153: (TYPE_NORMAL, 250), # Explosion
    154: (TYPE_NORMAL, 36),  # Fury Swipes
    155: (TYPE_GROUND, 100), # Bonemerang (2 hits)
    156: (TYPE_PSYCHIC, 0),  # Rest
    157: (TYPE_ROCK, 75),    # Rock Slide
    158: (TYPE_NORMAL, 80),  # Hyper Fang
    162: (TYPE_NORMAL, 50),  # Super Fang
    163: (TYPE_NORMAL, 70),  # Slash
    172: (TYPE_FIRE, 60),    # Flame Wheel
    188: (TYPE_POISON, 90),  # Sludge Bomb
    189: (TYPE_GROUND, 20),  # Mud-Slap
    200: (TYPE_DRAGON, 90),  # Outrage
    202: (TYPE_GRASS, 60),   # Giga Drain
    209: (TYPE_ELECTRIC, 65),# Spark
    210: (TYPE_BUG, 20),     # Fury Cutter
    211: (TYPE_STEEL, 70),   # Steel Wing
    228: (TYPE_DARK, 40),    # Pursuit
    231: (TYPE_STEEL, 100),  # Iron Tail
    232: (TYPE_STEEL, 50),   # Metal Claw
    233: (TYPE_FIGHTING, 70),# Vital Throw
    237: (TYPE_NORMAL, 50),  # Hidden Power
    240: (TYPE_WATER, 0),    # Rain Dance
    241: (TYPE_FIRE, 0),     # Sunny Day
    242: (TYPE_DARK, 80),    # Crunch
    247: (TYPE_GHOST, 80),   # Shadow Ball
    249: (TYPE_FIGHTING, 20),# Rock Smash
    252: (TYPE_NORMAL, 40),  # Fake Out
    257: (TYPE_FIRE, 100),   # Heat Wave
    263: (TYPE_NORMAL, 70),  # Facade
    264: (TYPE_FIGHTING, 150),# Focus Punch
    276: (TYPE_FIGHTING, 120),# Superpower
    279: (TYPE_FIGHTING, 60),# Revenge
    280: (TYPE_FIGHTING, 75),# Brick Break
    282: (TYPE_DARK, 20),    # Knock Off
    290: (TYPE_NORMAL, 70),  # Secret Power
    291: (TYPE_WATER, 60),   # Dive
    299: (TYPE_FIRE, 85),    # Blaze Kick
    315: (TYPE_FIRE, 140),   # Overheat
    317: (TYPE_ROCK, 50),    # Rock Tomb
    318: (TYPE_BUG, 60),     # Silver Wind
    324: (TYPE_BUG, 75),     # Signal Beam
    331: (TYPE_GRASS, 25),   # Bullet Seed
    332: (TYPE_FLYING, 60),  # Aerial Ace
    337: (TYPE_DRAGON, 80),  # Dragon Claw
    341: (TYPE_GROUND, 55),  # Mud Shot
    351: (TYPE_ELECTRIC, 60),# Shock Wave
    352: (TYPE_WATER, 60),   # Water Pulse
}


def type_multiplier(move_type: int, def_type1: int, def_type2: int) -> float:
    """Retorna o multiplicador de efetividade na Gen 3 (0.0, 0.25, 0.5, 1.0, 2.0 ou 4.0)."""
    chart = TYPE_CHART.get(move_type, {})
    m1 = chart.get(def_type1, 1.0)
    if def_type2 != def_type1:
        m2 = chart.get(def_type2, 1.0)
        return float(m1 * m2)
    return float(m1)


def score_moves(
    player_types: tuple[int, int],
    enemy_types: tuple[int, int],
    moves: tuple[int, ...],
    pp: tuple[int, ...],
) -> tuple[list[float], int, float]:
    """
    Calcula eficiência normalizada [0, 1] de cada golpe considerando base_power,
    STAB (1.5x), tabela de tipos Gen 3 e PP disponível.
    Retorna (scores[4], best_slot_idx, best_mult).
    """
    scores = [0.0, 0.0, 0.0, 0.0]
    mults = [1.0, 1.0, 1.0, 1.0]
    for i in range(min(4, len(moves), len(pp))):
        move_id = int(moves[i])
        cur_pp = int(pp[i])
        if move_id <= 0 or cur_pp <= 0:
            continue
        move_type, power = MOVES_DB.get(move_id, (TYPE_NORMAL, 40))
        if power <= 0:
            scores[i] = 0.05
            mults[i] = 1.0
            continue
        mult = type_multiplier(move_type, int(enemy_types[0]), int(enemy_types[1]))
        mults[i] = mult
        if mult == 0.0:
            scores[i] = 0.0
            continue
        stab = 1.5 if move_type in player_types else 1.0
        eff = power * stab * mult
        scores[i] = float(min(eff / 150.0, 1.0))

    best_idx = 0
    best_score = -1.0
    for i, sc in enumerate(scores):
        if sc > best_score:
            best_score = sc
            best_idx = i
    if best_score <= 0.0:
        return scores, 0, 1.0
    return scores, best_idx, mults[best_idx]
